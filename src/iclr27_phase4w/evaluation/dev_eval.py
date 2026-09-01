"""Q1 dev evaluation for Phase 4W state-conditioned TrackOCD.

Active known universe at dev = all 48 supported-known categories (true
novel categories are absent). Model-in-the-loop memory; frozen candidate
heads; dev GT used only for evaluation.
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
    known_evidence,
    load_known_branch,
    load_novel_branch,
    proto_evidence,
)
from src.iclr27_phase4w.cold_start.train import ColdStartHead, WarmMemoryHead
from src.iclr27_phase4w.episodes.build_episodes import load_active_universe
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
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--proposals", default=str(Q1_DEV))
    ap.add_argument("--feats", default="outputs/iclr27_phase4s/q1_features/feats.npz")
    ap.add_argument("--evidence-mode", choices=["active", "fixed48"],
                    default="active")
    ap.add_argument("--cold-head", default="outputs/iclr27_phase4w/cold_start/head_cold_v3/head.pth")
    ap.add_argument("--warm-head", default="outputs/iclr27_phase4w/warm_memory/head_warm_v3/head.pth")
    args = ap.parse_args()

    known_list = sorted(known_ids())
    ktsr, kcls = load_known_branch(args.device)
    ntsr, l2 = load_novel_branch(args.device)
    protos, cat_index, proj_t = load_active_universe(args.device)
    c = torch.load(ROOT / args.cold_head,
                   map_location=args.device)
    w = torch.load(ROOT / args.warm_head,
                   map_location=args.device)
    cold = ColdStartHead(c["dim"]).to(args.device)
    cold.load_state_dict(c["model"]); cold.eval()
    warm = WarmMemoryHead(w["dim"]).to(args.device)
    warm.load_state_dict(w["model"]); warm.eval()
    active_idx = list(range(len(known_list)))

    rows = load_proposals(Path(args.proposals))
    arr = np.load(ROOT / args.feats)["feats"]
    assert len(arr) == len(rows)
    qmap = qphys_from_rows(rows)
    feats_by_key = {}
    for i, r in enumerate(rows):
        feats_by_key[(int(r["video_id"]), int(r["track_id"]), int(r["image_id"]))] = arr[i]
    tracks = group_tracks(rows)
    wcoef = r_phys_calibration(rows)
    r_scalar = compute_r_phys(rows, wcoef)

    memory = NovelMemory(args.device)
    results = {}
    commits = []
    state_stats = defaultdict(int)
    with torch.no_grad():
        for key in sorted(tracks):
            ds = DualSpaceStep(ktsr, kcls, ntsr, l2, args.device)
            commit = None
            for t, r in enumerate(tracks[key]):
                f = feats_by_key.get((int(r["video_id"]), int(r["track_id"]),
                                      int(r["image_id"])))
                if f is None:
                    continue
                ft = torch.from_numpy(f).unsqueeze(0).to(args.device)
                qt = torch.tensor([qmap[id(r)]], device=args.device)
                rs = float(r_scalar[id(r)])
                ev, s_k, s_n, nl, l2_new = ds.step(ft, qt, rs, t + 1, memory)
                if args.evidence_mode == "active":
                    pe = proto_evidence(s_k, protos, active_idx, tau=0.1)
                else:
                    pe = known_evidence(kcls(s_k))
                skp = (torch.nn.functional.normalize(s_k, dim=-1) @ proj_t)[0]
                skp = skp.detach().cpu().numpy()
                qv = qt[0].cpu().numpy().astype(np.float32)
                phase_cold = memory.size() == 0
                if phase_cold:
                    x = np.concatenate([pe, skp, qv]).astype(np.float32)
                    logits = cold(torch.from_numpy(x).unsqueeze(0).to(args.device))[0]
                    a = int(logits.argmax())
                    if a == 0:
                        kl = kcls(s_k)[0]
                        commit = ("known", known_list[int(kl.argmax())], t)
                        state_stats["cold_known_commit"] += 1
                        break
                    elif a == 1:
                        commit = ("new", memory.size(), t)
                        memory.create(s_n, rs, {"track": key})
                        state_stats["cold_new_commit"] += 1
                        break
                    else:
                        continue
                else:
                    mem_ev = np.concatenate([ev[8:12], qv]).astype(np.float32)
                    x = np.concatenate([pe, skp, mem_ev]).astype(np.float32)
                    logits = warm(torch.from_numpy(x).unsqueeze(0).to(args.device))[0]
                    a = int(logits.argmax())
                    if a == 0:
                        kl = kcls(s_k)[0]
                        commit = ("known", known_list[int(kl.argmax())], t)
                        state_stats["warm_known_commit"] += 1
                        break
                    elif a == 1:
                        commit = ("new", memory.size(), t)
                        memory.create(s_n, rs, {"track": key})
                        state_stats["warm_new_commit"] += 1
                        break
                    elif a == 2:
                        if nl.shape[1] >= 1:
                            slot = int(nl.argmax())
                            commit = ("existing", slot, t)
                            memory.update(slot, s_n, rs)
                            state_stats["warm_existing_commit"] += 1
                        else:
                            commit = ("new", memory.size(), t)
                            memory.create(s_n, rs, {"track": key})
                            state_stats["warm_new_commit"] += 1
                        break
                    else:
                        continue
            results[key] = commit
            if commit is not None:
                commits.append((key, commit[0], commit[1], commit[2]))

    stream, labels = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)
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
    summary = {
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                    for k, v in metrics.items() if k != "hungarian_assignment"},
        "n_aligned": len(mapping),
        "memory_slots": memory.size(),
        "absorption": absorption,
        "state_stats": dict(state_stats),
        "r_phys_coefs": [float(x) for x in wcoef],
    }
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dev.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("metrics", "n_aligned", "memory_slots", "absorption",
                       "state_stats")}, indent=2))


if __name__ == "__main__":
    main()
