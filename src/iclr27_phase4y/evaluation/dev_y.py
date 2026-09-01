"""ADSSI Q1 dev evaluation (frozen candidate)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.dev_eval import compute_r_phys, r_phys_calibration
from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.protocol import (
    Q1_DEV,
    group_tracks,
    load_gt_tracks_dev,
    load_proposals,
)
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4w.evaluation.dev_eval import qphys_from_rows
from src.iclr27_phase4x.evaluation.pilot_x3 import load_tsr
from src.iclr27_phase4y.evaluation.rollout import run_track
from src.iclr27_phase4y.model import ADSSI, DynamicStateMemory
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--commit-threshold", type=float, default=0.5)
    ap.add_argument("--margin-ratio", type=float, default=1.5)
    ap.add_argument("--min-age", type=int, default=2)
    args = ap.parse_args()

    tsr = load_tsr(args.device)
    d = np.load(ROOT / "outputs/iclr27_phase4x/simple_mixture/known_anchors.npz")
    anchors = torch.from_numpy(d["means"]).to(args.device)
    cat_ids = d["cat_ids"].tolist()
    cat_index = {c: i for i, c in enumerate(cat_ids)}
    active_idx = list(range(len(cat_ids)))
    ck = torch.load(ROOT / args.checkpoint, map_location=args.device)
    model = ADSSI(in_dim=256, d=128).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    mem = DynamicStateMemory(model, anchors, args.device)

    rows = load_proposals(Path(Q1_DEV))
    arr = np.load(ROOT / "outputs/iclr27_phase4s/q1_features/feats.npz")["feats"]
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

    results = {}
    state_stats = defaultdict(int)
    with torch.no_grad():
        for key in sorted(tracks):
            zs, qs = [], []
            for r in tracks[key]:
                f = feats_by_key.get((int(r["video_id"]), int(r["track_id"]),
                                      int(r["image_id"])))
                if f is None:
                    continue
                zs.append(f)
                qs.append(qmap[id(r)])
            if not zs:
                results[key] = None
                continue
            z_seq = np.stack(zs).astype(np.float32)
            q_seq = np.stack(qs).astype(np.float32)
            a, t, scores, prop, s_commit = run_track(
                model, tsr, mem, z_seq, q_seq, args.min_age,
                args.commit_threshold, args.margin_ratio, args.device)
            if a is None:
                results[key] = None
                continue
            C = len(active_idx)
            if a < C:
                results[key] = ("known", cat_ids[a], t)
                state_stats["known_commit"] += 1
            elif a < C + mem.size():
                k = a - C
                results[key] = ("existing", k, t)
                zt = model.obs(s_commit)
                mem.update(k, zt, float(np.clip(q_seq[t, 0], 0.05, 0.95)))
                state_stats["existing_commit"] += 1
            elif a == C + mem.size():
                k = mem.create(prop, float(np.clip(q_seq[t, 0], 0.05, 0.95)))
                results[key] = ("new", k, t)
                state_stats["new_commit"] += 1
            else:
                results[key] = None
                state_stats["noise_commit"] += 1

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
    ev = TrackOCDEvaluator([labels[sid] for sid in sorted(labels)])
    metrics = ev.evaluate(preds, metadata={"memory_size": mem.size()})
    absorption = {"novel_gt": 0, "absorbed_as_known": 0, "routed_novel": 0}
    for key, sid in mapping.items():
        if labels[sid]["protocol_role"] != "novel":
            continue
        absorption["novel_gt"] += 1
        out = results[key]
        if out is not None and out[0] == "known":
            absorption["absorbed_as_known"] += 1
        elif out is not None and out[0] in ("new", "existing"):
            absorption["routed_novel"] += 1
    summary = {
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                    for k, v in metrics.items() if k != "hungarian_assignment"},
        "n_aligned": len(mapping),
        "memory_slots": mem.size(),
        "absorption": absorption,
        "state_stats": dict(state_stats),
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "dev.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
