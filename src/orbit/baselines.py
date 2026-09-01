"""Baseline ladder: trivial, online, and oracle diagnostics."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from src.dual_branch.memory.b2_adapter import B2Memory
from src.dual_branch.models.outputs import emit
from src.ocd_v2.common import load_train_known, build_prototypes
from src.orbit.protocol import load_mean_features, load_stream, load_gt, subset_ids
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def run_trivial(kind, rows, feats, protos, gt_rows, seed=1027):
    novel_ratio = sum(1 for g in gt_rows if g["protocol_role"] == "novel") / len(gt_rows)
    rng = random.Random(seed)
    preds = []
    if kind == "all_known" or kind == "nearest_known":
        for i, r in enumerate(rows):
            best_id, best_s = None, -1.0
            for cid, p in protos.items():
                s = float(np.dot(feats[r["sample_id"]], p))
                if s > best_s:
                    best_s, best_id = s, cid
            preds.append(emit(r["sample_id"], i, "known", known_id=best_id))
    elif kind == "all_novel":
        mem = B2Memory({}, threshold=0.45, novel_only=True)
        for i, r in enumerate(rows):
            vid, _ = mem.predict_one(feats[r["sample_id"]], r["sample_id"], i)
            preds.append(emit(r["sample_id"], i, "novel", virtual_id=vid))
    elif kind == "random_router":
        mem = B2Memory({}, threshold=0.45, novel_only=True)
        for i, r in enumerate(rows):
            if rng.random() < novel_ratio:
                vid, _ = mem.predict_one(feats[r["sample_id"]], r["sample_id"], i)
                preds.append(emit(r["sample_id"], i, "novel", virtual_id=vid))
            else:
                best_id, best_s = None, -1.0
                for cid, p in protos.items():
                    s = float(np.dot(feats[r["sample_id"]], p))
                    if s > best_s:
                        best_s, best_id = s, cid
                preds.append(emit(r["sample_id"], i, "known", known_id=best_id))
    return preds


def run_streaming(kind, rows, feats, protos, gt_rows, threshold=0.45):
    if kind == "streaming_centroid":
        mem = B2Memory(protos, threshold=threshold)
    elif kind == "dpmeans_like":
        mem = B2Memory(protos, threshold=0.3, ema=False)
    preds = []
    for i, r in enumerate(rows):
        vid, kind_ = mem.predict_one(feats[r["sample_id"]], r["sample_id"], i)
        preds.append(emit(r["sample_id"], i, kind_,
                          vid if kind_ == "known" else None,
                          vid if kind_ == "novel" else None))
    return preds


def run_oracle_routing(rows, feats, gt_rows, threshold=0.45):
    gt_by_sid = {g["sample_id"]: g for g in gt_rows}
    mem = B2Memory({}, threshold=threshold, novel_only=True)
    preds = []
    for i, r in enumerate(rows):
        g = gt_by_sid.get(r["sample_id"])
        if g is not None and g["protocol_role"] in ("supported_known", "zero_shot_known"):
            preds.append(emit(r["sample_id"], i, "known", known_id=g["ground_truth_category_id"]))
        else:
            vid, _ = mem.predict_one(feats[r["sample_id"]], r["sample_id"], i)
            preds.append(emit(r["sample_id"], i, "novel", virtual_id=vid))
    return preds


def run_oracle_known(rows, feats, gt_rows, protos, threshold=0.45):
    gt_by_sid = {g["sample_id"]: g for g in gt_rows}
    mem = B2Memory({}, threshold=threshold, novel_only=True)
    preds = []
    for i, r in enumerate(rows):
        g = gt_by_sid.get(r["sample_id"])
        if g is not None and g["protocol_role"] in ("supported_known", "zero_shot_known"):
            preds.append(emit(r["sample_id"], i, "known", known_id=g["ground_truth_category_id"]))
        else:
            vid, _ = mem.predict_one(feats[r["sample_id"]], r["sample_id"], i)
            preds.append(emit(r["sample_id"], i, "novel", virtual_id=vid))
    return preds


def run_offline_kmeans(rows, feats, gt_rows, known_cats, seed=1027):
    X = np.stack([feats[r["sample_id"]] for r in rows])
    cats = sorted({g["ground_truth_category_id"] for g in gt_rows
                   if g["protocol_role"] != "distractor"})
    km = KMeans(n_clusters=len(cats), n_init=5, random_state=seed, max_iter=300).fit(X)
    W = np.zeros((len(cats), len(cats)), dtype=int)
    for i, r in enumerate(rows):
        g = next(g for g in gt_rows if g["sample_id"] == r["sample_id"])
        W[km.labels_[i], cats.index(g["ground_truth_category_id"])] += 1
    rows_, cols_ = linear_sum_assignment(-W)
    mapping = {int(rows_[i]): cats[int(cols_[i])] for i in range(len(rows_))}
    preds = []
    for i, r in enumerate(rows):
        c = mapping[int(km.labels_[i])]
        if c in known_cats:
            preds.append(emit(r["sample_id"], i, "known", known_id=c))
        else:
            preds.append(emit(r["sample_id"], i, "novel", virtual_id=int(km.labels_[i])))
    return preds


def main():
    val_feats = load_mean_features("gt_tracks_mean")
    tr_feats, labels = load_train_known("dinov2")
    protos = build_prototypes(tr_feats, labels, set(labels.values()))
    known_cats = set(labels.values())
    gt = load_gt("pure")
    rows = load_stream("pure", "main_seed1027")
    methods = ["all_known", "nearest_known", "all_novel", "random_router",
               "streaming_centroid", "dpmeans_like", "trackocd_ref",
               "oracle_routing", "oracle_known", "offline_kmeans_oracle_k"]
    out_rows = []
    for m in methods:
        if m == "trackocd_ref":
            mem = B2Memory(protos, threshold=0.45)
            preds = []
            for i, r in enumerate(rows):
                vid, kind = mem.predict_one(val_feats[r["sample_id"]], r["sample_id"], i)
                preds.append(emit(r["sample_id"], i, kind,
                                  vid if kind == "known" else None,
                                  vid if kind == "novel" else None))
        elif m in ("all_known", "nearest_known", "all_novel", "random_router"):
            preds = run_trivial(m, rows, val_feats, protos, gt)
        elif m in ("streaming_centroid", "dpmeans_like"):
            preds = run_streaming(m, rows, val_feats, protos, gt)
        elif m == "oracle_routing":
            preds = run_oracle_routing(rows, val_feats, gt)
        elif m == "oracle_known":
            preds = run_oracle_known(rows, val_feats, gt, protos)
        elif m == "offline_kmeans_oracle_k":
            preds = run_offline_kmeans(rows, val_feats, gt, known_cats)
        ev = TrackOCDEvaluator(gt)
        res = ev.evaluate(preds)
        row = {
            "Method": m, "Track input": "GT tracks", "Online causal": m not in ("offline_kmeans_oracle_k",),
            "Uses future tracks": False, "Oracle routing": m in ("oracle_routing", "oracle_known"),
            "Oracle K": m == "offline_kmeans_oracle_k",
            "Known ACC": res["overall_known_acc"], "RN-Acc": res["route_aware_novel_acc"],
            "Conditional Novel ACC": res["conditional_novel_acc"], "Routing Recall": res["novel_routing_recall"],
            "NMI": res["novel_only_nmi"], "ARI": res["novel_only_ari"],
            "Count Error": res["novel_count_abs_error"], "All ACC": res["all_track_acc"],
        }
        out_rows.append(row)
        print(row)
    out = ROOT / "outputs" / "orbit" / "baselines" / "baseline_ladder.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)


if __name__ == "__main__":
    main()
