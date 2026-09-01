"""Q1 dev evaluation for DS-TrackOCD (frozen branches + independent router)."""
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
    load_known_branch,
    load_novel_branch,
)
from src.iclr27_phase4v.train_router import LogisticRouter, MLPRouter
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
    ap.add_argument("--router", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--proposals", default=str(Q1_DEV))
    ap.add_argument("--feats", default="outputs/iclr27_phase4s/q1_features/feats.npz")
    args = ap.parse_args()

    known_list = sorted(known_ids())
    ktsr, kcls = load_known_branch(args.device)
    ntsr, l2 = load_novel_branch(args.device)
    rc = torch.load(ROOT / args.router, map_location=args.device)
    dim = int(rc.get("dim", 15))
    router = (LogisticRouter(dim) if rc["arch"] == "logistic"
              else MLPRouter(dim)).to(args.device)
    router.load_state_dict(rc["model"])
    router.eval()

    rows = load_proposals(Path(args.proposals))
    arr = np.load(ROOT / args.feats)["feats"]
    assert len(arr) == len(rows)
    qmap = qphys_from_rows(rows)
    feats_by_key = {}
    for i, r in enumerate(rows):
        feats_by_key[(int(r["video_id"]), int(r["track_id"]), int(r["image_id"]))] = arr[i]
    tracks = group_tracks(rows)
    w = r_phys_calibration(rows)
    r_scalar = compute_r_phys(rows, w)

    memory = NovelMemory(args.device)
    results = {}
    commits = []
    with torch.no_grad():
        for key in sorted(tracks):
            ds = DualSpaceStep(ktsr, kcls, ntsr, l2, args.device)
            first_commit = None
            known_prob = None
            for t, r in enumerate(tracks[key]):
                f = feats_by_key.get((int(r["video_id"]), int(r["track_id"]),
                                      int(r["image_id"])))
                if f is None:
                    continue
                ft = torch.from_numpy(f).unsqueeze(0).to(args.device)
                qt = torch.tensor([qmap[id(r)]], device=args.device)
                rs = float(r_scalar[id(r)])
                ev, s_k, s_n, nl, l2_new = ds.step(ft, qt, rs, t + 1, memory)
                r_logits = router(torch.from_numpy(ev).unsqueeze(0).to(args.device))[0]
                rp = torch.softmax(r_logits, dim=-1)[1].item()
                if r_logits[1] > r_logits[0]:  # KNOWN
                    if first_commit is None:
                        kl = kcls(s_k)[0]
                        pred = known_list[int(kl.argmax())]
                        first_commit = ("known", pred, t)
                        known_prob = rp
                        commits.append((key, "known", pred, t))
                else:  # NOVEL
                    if first_commit is None:
                        if nl.shape[1] >= 1 and nl.max() >= l2_new[0, 0]:
                            slot = int(nl.argmax())
                            first_commit = ("existing", slot, t)
                            known_prob = rp
                            memory.update(slot, s_n, rs)
                            commits.append((key, "existing", slot, t))
                        else:
                            first_commit = ("new", memory.size(), t)
                            known_prob = rp
                            memory.create(s_n, rs, {"track": key})
                            commits.append((key, "new", memory.size(), t))
            results[key] = {"outcome": first_commit,
                            "router_known_prob": known_prob}

    stream, labels = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)
    preds = []
    order = 0
    for key, sid in mapping.items():
        order += 1
        out = results[key]["outcome"]
        if out is None or out[0] == "defer":
            preds.append({"sample_id": sid, "prediction_type": "unresolved",
                          "stream_order": order})
        elif out[0] == "known":
            preds.append({"sample_id": sid, "prediction_type": "known",
                          "semantic_category_id": out[1], "stream_order": order})
        else:
            preds.append({"sample_id": sid, "prediction_type": "novel",
                          "virtual_category_id": out[1], "stream_order": order})
    ev = TrackOCDEvaluator([labels[sid] for sid in sorted(labels)])
    metrics = ev.evaluate(preds, metadata={"memory_size": memory.size()})

    # absorption analysis on aligned GT novel tracks
    absorption = {"novel_gt": 0, "absorbed_as_known": 0, "routed_novel": 0}
    for key, sid in mapping.items():
        lab = labels[sid]
        if lab["protocol_role"] != "novel":
            continue
        absorption["novel_gt"] += 1
        out = results[key]["outcome"]
        if out is not None and out[0] == "known":
            absorption["absorbed_as_known"] += 1
        elif out is not None and out[0] in ("existing", "new"):
            absorption["routed_novel"] += 1

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                    for k, v in metrics.items() if k != "hungarian_assignment"},
        "n_aligned": len(mapping),
        "memory_slots": memory.size(),
        "absorption": absorption,
        "router_known_prob_mean": float(np.mean([
            v["router_known_prob"] for v in results.values()
            if v["router_known_prob"] is not None])) if any(
                v["router_known_prob"] is not None for v in results.values()) else None,
    }
    (out_dir / "dev.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
