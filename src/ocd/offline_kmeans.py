#!/usr/bin/env python3
"""B0 / B1 offline K-means diagnostics (Oracle-K upper bound, NOT OCD)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_predictions, load_private_labels


def load_track_features(cache_dir):
    feats = {}
    for p in cache_dir.glob("*.json"):
        r = json.loads(p.read_text())
        feats[r["sample_id"]] = np.asarray(r["mean_embedding"], dtype=np.float32)
    return feats


def run(encoder, mode, subset):
    if mode == "single":
        cache_dir = PROJECT_ROOT / "data" / "caches" / "features" / encoder / "gt_tracks_single"
    else:
        cache_dir = PROJECT_ROOT / "data" / "caches" / "features" / encoder / "gt_tracks_mean"
    feats = load_track_features(cache_dir)
    labels = load_private_labels(PROJECT_ROOT)

    ids = sorted(feats.keys())
    X = np.stack([feats[i] for i in ids])
    y = np.array([labels[i]["ground_truth_category_id"] for i in ids])
    known = np.array([labels[i]["is_known"] for i in ids])

    unk_mask = ~known
    Xu = X[unk_mask]
    yu = y[unk_mask]
    uids = [ids[j] for j in np.where(unk_mask)[0]]

    if subset == "full":
        pass
    else:
        manifest = PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "manifests" / f"{subset}_track_ids.json"
        keep_ids = set(json.loads(manifest.read_text()))
        keep = np.array([u in keep_ids for u in uids])
        Xu, yu, uids = Xu[keep], yu[keep], [uids[j] for j in np.where(keep)[0]]

    K = len(set(yu.tolist()))
    km = KMeans(n_clusters=K, n_init=10, random_state=1027)
    pred = km.fit_predict(Xu)
    # renumber virtual ids so they don't collide with known ids in evaluation
    pred_virtual = pred + 100000
    known_mask = np.zeros(len(yu), dtype=bool)
    res = evaluate_predictions(yu, pred_virtual, known_mask)
    res["method"] = f"kmeans_{mode}"
    res["encoder"] = encoder
    res["subset"] = subset
    res["K_oracle"] = K
    res["sample_ids"] = uids
    res["cluster_ids"] = pred.tolist()

    out_json = PROJECT_ROOT / "runs" / "gt_kmeans" / f"{encoder}_{mode}_{subset}_assignments.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1, default=str))
    csv_path = PROJECT_ROOT / "outputs" / "metrics" / "gt_kmeans.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a") as f:
        f.write(
            f"kmeans_{mode},{encoder},{subset},oracle,"
            f"{res['acc_all']:.4f},{res['acc_known']:.4f},{res['acc_novel']:.4f},"
            f"{res['nmi']:.4f},{res['ari']:.4f},{res['predicted_categories']},"
            f"{res['category_count_abs_error']},{res['mean_fragmentation']:.4f},"
            f"{res['merge_error']:.4f},{res['duplicate_creation_rate']:.4f}\n"
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["dinov2", "clip"], required=True)
    ap.add_argument("--mode", choices=["single", "mean"], required=True)
    ap.add_argument("--subset", choices=["full", "repeated", "balanced"], default="full")
    args = ap.parse_args()
    res = run(args.encoder, args.mode, args.subset)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
