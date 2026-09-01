"""External online baseline: frozen TrackOCD-Ref router + pdc-DP-means.

Known routing uses the frozen TrackOCD-Ref B2 memory (DINO track means,
threshold 0.45).  Novel-routed tracks are clustered online with the official
MiniBatchDPMeans.partial_fit (pinned repo, pdc-dp-means 0.0.8).  delta is
selected on the train-side meta-dev proxy only.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pdc_dp_means import MiniBatchDPMeans

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import (
    load_mean_features,
    load_stream,
    load_gt,
    load_train_labels,
    meta_classes,
)
from src.dual_branch.memory.b2_adapter import B2Memory
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def build_protos(class_ids):
    feats = load_mean_features("train_known_mean")
    labels = load_train_labels()
    sums = defaultdict(lambda: np.zeros(768, dtype=np.float32))
    counts = defaultdict(int)
    for sid, c in labels.items():
        if c in class_ids and sid in feats:
            sums[c] += feats[sid]
            counts[c] += 1
    protos = {}
    for c, s in sums.items():
        v = s / counts[c]
        protos[c] = v / (np.linalg.norm(v) + 1e-12)
    return protos


def run_stream(rows, feats, protos, delta):
    b2 = B2Memory(protos, threshold=0.45)
    dp = MiniBatchDPMeans(n_clusters=1, delta=delta, random_state=0,
                          compute_labels=True)
    preds = []
    novel_count = 0
    for i, r in enumerate(rows):
        z = feats[r["sample_id"]]
        vid, kind = b2.predict_one(z, r["sample_id"], i)
        if kind == "known":
            preds.append({"sample_id": r["sample_id"], "stream_order": i,
                          "prediction_type": "known",
                          "semantic_category_id": int(vid)})
        else:
            dp.partial_fit(z.reshape(1, -1))
            label = int(dp.predict(z.reshape(1, -1))[0])
            preds.append({"sample_id": r["sample_id"], "stream_order": i,
                          "prediction_type": "novel",
                          "virtual_category_id": 100000 + label})
            novel_count += 1
    return preds, novel_count, len(dp.cluster_centers_)


def meta_dev_deltas():
    """Train-side delta selection on the meta-dev proxy."""
    labels = load_train_labels()
    meta_tr = meta_classes("meta_train_classes")
    meta_dev = meta_classes("meta_dev_classes")
    feats = load_mean_features("train_known_mean")
    protos = build_protos(meta_tr)
    dev_ids = [sid for sid, c in labels.items() if c in meta_dev and sid in feats]
    known_ids = [sid for sid, c in labels.items() if c in meta_tr and sid in feats]
    rows = [{"sample_id": sid} for sid in sorted(dev_ids)] + \
           [{"sample_id": sid} for sid in sorted(known_ids)[:600]]
    gt = [{"sample_id": sid, "ground_truth_category_id": labels[sid],
           "protocol_role": "novel" if labels[sid] in meta_dev else "supported_known"}
          for sid in (sorted(dev_ids) + sorted(known_ids)[:600])]
    out = []
    for delta in [0.9, 1.2]:
        preds, n_novel, n_centers = run_stream(rows, feats, protos, delta)
        ev = TrackOCDEvaluator(gt)
        res = ev.evaluate(preds)
        row = {"delta": delta,
               "known_acc": res["overall_known_acc"],
               "rn_acc": res["route_aware_novel_acc"],
               "cond_novel_acc": res["conditional_novel_acc"],
               "routing_recall": res["novel_routing_recall"],
               "nmi": res["novel_only_nmi"], "ari": res["novel_only_ari"],
               "count_error": res["novel_count_abs_error"],
               "predicted_novel_count": res["predicted_novel_count"]}
        out.append(row)
        print(row, flush=True)
    return out


def official(delta):
    gt = load_gt("pure")
    rows = load_stream("pure", "main_seed1027")
    feats = load_mean_features("gt_tracks_mean")
    protos = build_protos(set(load_train_labels().values()))
    preds, n_novel, n_centers = run_stream(rows, feats, protos, delta)
    ev = TrackOCDEvaluator(gt)
    res = ev.evaluate(preds)
    return res, preds, n_centers


def main():
    out_dir = ROOT / "outputs" / "iclr27_phase4d" / "external_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = meta_dev_deltas()
    with open(out_dir / "pdc_dp_means_meta_dev.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # select delta by ARI then count error (train-side only)
    best = max(rows, key=lambda r: (r["ari"], -r["count_error"]))
    delta = best["delta"]
    print("selected delta", delta)
    res, preds, n_centers = official(delta)
    row = {"delta": delta, "all_acc": res["all_track_acc"],
           "known_acc": res["overall_known_acc"],
           "rn_acc": res["route_aware_novel_acc"],
           "cond_novel_acc": res["conditional_novel_acc"],
           "routing_recall": res["novel_routing_recall"],
           "nmi": res["novel_only_nmi"], "ari": res["novel_only_ari"],
           "count_error": res["novel_count_abs_error"],
           "predicted_novel_count": res["predicted_novel_count"],
           "dp_centers": n_centers}
    with open(out_dir / "pdc_dp_means_seed1027.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    (ROOT / "runs/iclr27_phase4d/pdc_dp_means_seed1027.json").write_text(
        json.dumps({**res, "prediction_log": preds}, indent=1, default=str))
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
