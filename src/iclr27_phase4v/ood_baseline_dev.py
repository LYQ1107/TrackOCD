"""Simple OOD-router baselines on Q1 dev (V1 in the Phase 4V matrix).

Router = a fixed threshold on one evidence statistic from the frozen known
branch (energy / prototype energy / entropy / top1 / new_logit). NO
episodic router training. Dev GT is used only for evaluation; threshold
sweeps are reported as a diagnostic frontier, not as a tuned final method.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.dev_eval import compute_r_phys, r_phys_calibration
from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.model import NovelMemory
from src.iclr27_phase4s.protocol import (
    Q1_DEV,
    group_tracks,
    known_ids,
    load_gt_tracks_dev,
    load_proposals,
)
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4v.evidence import (
    DualSpaceStep,
    build_known_protos,
    load_known_branch,
    load_novel_branch,
    proto_evidence,
)
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def qphys_from_rows(rows):
    by_track = defaultdict(list)
    for r in rows:
        by_track[(r["video_id"], r["track_id"])].append(r)
    out = {}
    for key, idxs in by_track.items():
        idxs.sort(key=lambda r: (r["frame_id"], int(r.get("proposal_local_id") or 0)))
        last_frame, hits, ssum, n = None, 0, 0.0, 0
        for r in idxs:
            gap = 0 if last_frame is None else r["frame_id"] - last_frame - 1
            b = json.loads(r["bbox_xyxy"])
            area = max(b[2] - b[0], 1) * max(b[3] - b[1], 1)
            out[id(r)] = [r["score"], float(np.log1p(hits)), min(hits, 16) / 16.0,
                          float(np.log1p(max(gap, 0))),
                          ssum / n if n else r["score"],
                          float(np.log(area) / 12.0)]
            last_frame = r["frame_id"]
            hits += 1
            ssum += r["score"]
            n += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence", default="cls_energy",
                    choices=["cls_energy", "proto_energy", "proto_top1",
                             "proto_entropy", "new_logit", "q_run"])
    ap.add_argument("--min-age", type=int, default=2)
    ap.add_argument("--thresholds", default="4.5,5.0,5.25,5.5,5.75,6.0,6.25,6.5")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]

    known_list = sorted(known_ids())
    ktsr, kcls = load_known_branch(args.device)
    ntsr, l2 = load_novel_branch(args.device)
    protos = build_known_protos(ktsr, args.device).to(args.device)
    rows = load_proposals(Path(Q1_DEV))
    arr = np.load(ROOT / "outputs/iclr27_phase4s/q1_features/feats.npz")["feats"]
    assert len(arr) == len(rows)
    qmap = qphys_from_rows(rows)
    feats_by_key = {}
    for i, r in enumerate(rows):
        feats_by_key[(int(r["video_id"]), int(r["track_id"]), int(r["image_id"]))] = arr[i]
    tracks = group_tracks(rows)
    w = r_phys_calibration(rows)
    r_scalar = compute_r_phys(rows, w)

    stream, labels = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)

    def stat(run, ev, s_k):
        if args.evidence == "cls_energy":
            return float(ev[7])
        if args.evidence == "proto_energy":
            return float(proto_evidence(s_k, protos)[3])
        if args.evidence == "proto_top1":
            return float(proto_evidence(s_k, protos)[0])
        if args.evidence == "proto_entropy":
            return float(proto_evidence(s_k, protos)[2])
        if args.evidence == "new_logit":
            return float(ev[10])
        if args.evidence == "q_run":
            return float(ev[17])
        raise ValueError(run)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"evidence": args.evidence, "min_age": args.min_age, "rows": {}}
    for threshold in thresholds:
        memory = NovelMemory(args.device)
        results = {}
        with torch.no_grad():
            for key in sorted(tracks):
                ds = DualSpaceStep(ktsr, kcls, ntsr, l2, args.device)
                commit = None
                for t, r in enumerate(tracks[key]):
                    f = feats_by_key.get((int(r["video_id"]), int(r["track_id"]),
                                          int(r["image_id"])))
                    if f is None:
                        continue
                    if t < args.min_age:
                        continue
                    ft = torch.from_numpy(f).unsqueeze(0).to(args.device)
                    qt = torch.tensor([qmap[id(r)]], device=args.device)
                    rs = float(r_scalar[id(r)])
                    ev, s_k, s_n, nl, l2_new = ds.step(ft, qt, rs, t + 1, memory)
                    v = stat(None, ev, s_k)
                    novel = v < threshold  # low score -> novel for all stats here
                    if novel:
                        if nl.shape[1] >= 1 and nl.max() >= l2_new[0, 0]:
                            slot = int(nl.argmax())
                            commit = ("existing", slot, t)
                            memory.update(slot, s_n, rs)
                        else:
                            commit = ("new", memory.size(), t)
                            memory.create(s_n, rs, {"track": key})
                    else:
                        kl = kcls(s_k)[0]
                        commit = ("known", known_list[int(kl.argmax())], t)
                    break
                results[key] = commit
        preds = []
        order = 0
        for key, sid in mapping.items():
            order += 1
            out = results[key]
            if out is None:
                preds.append({"sample_id": sid, "prediction_type": "unresolved",
                              "stream_order": order})
            elif out[0] == "known":
                preds.append({"sample_id": sid, "prediction_type": "known",
                              "semantic_category_id": out[1], "stream_order": order})
            else:
                preds.append({"sample_id": sid, "prediction_type": "novel",
                              "virtual_category_id": out[1], "stream_order": order})
        ev_m = TrackOCDEvaluator([labels[sid] for sid in sorted(labels)])
        metrics = ev_m.evaluate(preds, metadata={"memory_size": memory.size()})
        absorption = {"novel_gt": 0, "absorbed_as_known": 0, "routed_novel": 0}
        for key, sid in mapping.items():
            if labels[sid]["protocol_role"] != "novel":
                continue
            absorption["novel_gt"] += 1
            out = results[key]
            if out is not None and out[0] == "known":
                absorption["absorbed_as_known"] += 1
            elif out is not None:
                absorption["routed_novel"] += 1
        summary["rows"][str(threshold)] = {
            "supported_known_acc": float(metrics["supported_known_acc"]),
            "novel_routing_recall": float(metrics["novel_routing_recall"]),
            "route_aware_novel_acc": float(metrics["route_aware_novel_acc"]),
            "count_abs_error": float(metrics["novel_count_abs_error"]),
            "memory_slots": memory.size(),
            "absorption": absorption,
            "harmonic": float(metrics["macro_known_novel_harmonic"]),
        }
        print(threshold, summary["rows"][str(threshold)], flush=True)
    (out_dir / "baseline.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
