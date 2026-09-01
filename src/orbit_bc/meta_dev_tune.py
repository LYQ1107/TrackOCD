"""Train-side meta-dev comparison for ORBIT-BC birth thresholds."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.evaluate import load_model, build_known
from src.orbit.protocol import load_frame_features, load_train_labels, meta_classes
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
from src.orbit_bc.evaluate_bc import run_stream_bc


def proxy_rows():
    train_feats = load_frame_features("train_known_mean")
    train_labels = load_train_labels()
    meta_tr = meta_classes("meta_train_classes")
    meta_dev = meta_classes("meta_dev_classes")
    dev_ids = sorted(sid for sid, c in train_labels.items() if c in meta_dev and sid in train_feats)
    rows = [{"sample_id": sid, "stream_order": i} for i, sid in enumerate(dev_ids)]
    gt = [{"sample_id": sid, "ground_truth_category_id": train_labels[sid],
           "protocol_role": "novel"} for sid in dev_ids]
    return train_feats, train_labels, meta_tr, rows, gt


def main():
    device = "cuda"
    model, _ = load_model(ROOT / "runs/orbit/model_D1_b128_g0.3/model.pth", device=device)
    train_feats, train_labels, meta_tr, rows, gt = proxy_rows()
    protos, radii = build_known(model, train_feats, train_labels, meta_tr, device)
    results = []
    for thr in [0.0, 0.45, 0.55, 0.65, 0.75]:
        preds, _ = run_stream_bc(model, rows, train_feats, protos, radii, device,
                                 birth_threshold=thr)
        ev = TrackOCDEvaluator(gt)
        r = ev.evaluate(preds)
        results.append({
            "birth_threshold": thr,
            "all_acc": r["all_track_acc"],
            "known_acc": r["overall_known_acc"],
            "rn_acc": r["route_aware_novel_acc"],
            "cond_novel_acc": r["conditional_novel_acc"],
            "routing_recall": r["novel_routing_recall"],
            "nmi": r["novel_only_nmi"],
            "ari": r["novel_only_ari"],
            "count_error": r["novel_count_abs_error"],
            "predicted_novel_count": r["predicted_novel_count"],
        })
        print(results[-1], flush=True)
    out = ROOT / "outputs/orbit_bc/meta_dev/config_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)


if __name__ == "__main__":
    main()
