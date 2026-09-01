#!/usr/bin/env python3
"""Part II: rerun TrackOCD baselines B0-B5 under the corrected
TrackOCD-v1.0 evaluator, for Pure and OV-assisted protocols.

B0/B1 are offline Oracle-K K-means diagnostics (GT labels used only for the
cluster->category mapping; clearly not OCD). B2/B3/B4 reuse the exact legacy
prediction logs (same stream order, same seeds). B5 is the oracle-gate
diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import hungarian_acc
from src.ocd_v2.common import load_mean_features
from src.ocd_v2.online_clustering import MultiPrototypeMemory
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator

DATA = PROJECT_ROOT / "data" / "trackocd_v1"
LEGACY_DATA = PROJECT_ROOT / "data" / "tao_ow_ocd_v1"
OUT = PROJECT_ROOT / "outputs" / "trackocd_v1" / "metrics"
RUNS = PROJECT_ROOT / "runs" / "trackocd_v1"

PROTOCOLS = ["pure", "ov_assisted"]
SUBSETS = ["full", "repeated", "balanced"]
STREAMS = ["main", "main_seed1027", "main_seed1028", "main_seed1029"]
SEED_STREAMS = ["main_seed1027", "main_seed1028", "main_seed1029"]


def load_gt(proto):
    rows = []
    with open(DATA / proto / "private" / "val_gt_track_labels.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def subset_ids(proto, subset):
    return set(json.loads((DATA / proto / "splits" / f"{subset}_track_ids.json").read_text()))


def stream_orders():
    orders = {}
    for st in STREAMS:
        rows = []
        fname = "val_gt_track_stream.jsonl" if st == "main" else f"val_gt_track_stream_{st[5:]}.jsonl"
        with open(LEGACY_DATA / "public" / fname) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        orders[st] = {r["sample_id"]: i for i, r in enumerate(rows)}
    return orders


def to_preds(log, id_map=None, ceiling=100000):
    preds = []
    for e in log:
        v = int(e["virtual_category_id"])
        base = {
            "sample_id": e["sample_id"],
            "stream_order": int(e.get("stream_order", 0)),
        }
        if v < ceiling:
            sem = id_map[v] if id_map is not None else v
            preds.append({**base, "prediction_type": "known", "semantic_category_id": int(sem)})
        else:
            preds.append({**base, "prediction_type": "novel", "virtual_category_id": v})
    return preds


def kmeans_oracle_k(feats, order, subset, gt_rows, k, seed, known_cats):
    from sklearn.cluster import KMeans
    from scipy.optimize import linear_sum_assignment
    sid_order = sorted(subset, key=lambda s: order[s])
    X = np.stack([feats[s] for s in sid_order])
    km = KMeans(n_clusters=k, n_init=5, random_state=seed, max_iter=300).fit(X)
    labels = km.labels_
    gt_by_sid = {r["sample_id"]: r for r in gt_rows}
    cats = sorted({gt_by_sid[s]["ground_truth_category_id"] for s in sid_order})
    W = np.zeros((k, len(cats)), dtype=np.int64)
    for sid, lab in zip(sid_order, labels):
        W[lab, cats.index(gt_by_sid[sid]["ground_truth_category_id"])] += 1
    rows_, cols_ = linear_sum_assignment(-W)
    cluster_to_cat = {int(rows_[i]): cats[int(cols_[i])] for i in range(len(rows_))}
    preds = []
    for i, sid in enumerate(sid_order):
        cat = cluster_to_cat[int(labels[i])]
        if cat in known_cats:
            preds.append({
                "sample_id": sid, "stream_order": order[sid],
                "prediction_type": "known", "semantic_category_id": cat,
            })
        else:
            preds.append({
                "sample_id": sid, "stream_order": order[sid],
                "prediction_type": "novel", "virtual_category_id": int(labels[i]),
            })
    return preds


def run_b0_b1(proto, subset, stream, order, feats, gt_rows, known_cats, seed_int):
    sub = subset_ids(proto, subset)
    cats = {
        r["ground_truth_category_id"] for r in gt_rows
        if r["sample_id"] in sub and r["protocol_role"] != "distractor"
    }
    return kmeans_oracle_k(feats, order, sub, gt_rows, len(cats), seed_int, known_cats)


def run_b2(subset, stream):
    p = PROJECT_ROOT / "runs" / "gt_online_ncm" / f"dinov2_{subset}_{stream}.json"
    r = json.loads(p.read_text())
    return to_preds(r["prediction_log"])


def phe_checkpoint(seed):
    return torch_load(PROJECT_ROOT / "runs" / "phe_track" / f"dinov2_seed{seed}" / "checkpoint.pth")


def run_b3(subset, stream, seed, orders):
    import torch
    from src.ocd.phe_track.phe_track_model import PPNetTrack
    from src.ocd.phe_track.eval_phe_track import class_hash_center, hash_code, simulate

    suffix = "" if stream == "main" else f"_{stream}"
    legacy_path = PROJECT_ROOT / "runs" / "gt_phe_track" / f"dinov2_seed{seed}_full_main{suffix}.json"
    if subset == "full" and legacy_path.exists():
        r = json.loads(legacy_path.read_text())
        log = r["prediction_log"]
        radius = int(r["radius"])
    else:
        ck = torch.load(PROJECT_ROOT / "runs" / "phe_track" / f"dinov2_seed{seed}" / "checkpoint.pth", map_location="cpu")
        radius_path = PROJECT_ROOT / "runs" / "gt_phe_track" / f"dinov2_seed{seed}_full_main.json"
        radius = int(json.loads(radius_path.read_text())["radius"])
        model = PPNetTrack(
            in_dim=768, prototype_dim=768,
            num_classes=len(ck["class_ids"]), global_proto_per_class=10,
            hash_code_length=12,
        )
        model.load_state_dict(ck["ema"])
        model.cuda().eval()
        feats = load_mean_features("dinov2", "gt_tracks_mean")
        stream_rows = []
        fname = "val_gt_track_stream.jsonl" if stream == "main" else f"val_gt_track_stream_{stream[5:]}.jsonl"
        with open(LEGACY_DATA / "public" / fname) as f:
            for line in f:
                if line.strip():
                    stream_rows.append(json.loads(line))
        known_centers = [
            (i, class_hash_center(model, i)) for i in range(len(ck["class_ids"]))
        ]
        _, log = simulate(model, stream_rows, feats, known_centers, radius)
    ck = torch_load(PROJECT_ROOT / "runs" / "phe_track" / f"dinov2_seed{seed}" / "checkpoint.pth")
    id_map = {i: int(c) for i, c in enumerate(ck["class_ids"])}
    return to_preds(log, id_map=id_map)


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu")


def run_b4(subset, stream):
    p = PROJECT_ROOT / "runs" / "arch1_5" / f"ocd_v2_dual_{stream}_{subset}.json"
    r = json.loads(p.read_text())
    return to_preds(r["prediction_log"], ceiling=200000)


def run_b5(subset, stream, order, feats, gt_rows):
    params = {
        "attach_thr": 0.525,
        "create_thr": 0.375,
        "new_proto_thr": 0.475,
        "max_proto": 4,
        "ema": 0.25,
        "maturity_tracks": 2,
    }
    stream_rows = []
    fname = "val_gt_track_stream.jsonl" if stream == "main" else f"val_gt_track_stream_{stream[5:]}.jsonl"
    with open(LEGACY_DATA / "public" / fname) as f:
        for line in f:
            if line.strip():
                stream_rows.append(json.loads(line))
    gt_by_sid = {r["sample_id"]: r for r in gt_rows}
    model = MultiPrototypeMemory(**params)
    preds = []
    for i, row in enumerate(stream_rows):
        sid = row["sample_id"]
        g = gt_by_sid.get(sid)
        if g is None or g["protocol_role"] == "distractor":
            continue
        if g["protocol_role"] in ("supported_known", "zero_shot_known"):
            preds.append({
                "sample_id": sid, "stream_order": i,
                "prediction_type": "known", "semantic_category_id": g["ground_truth_category_id"],
            })
        else:
            vid = model.predict_one(
                feats[sid], sid, i,
                num_frames=len(row.get("frame_ids", []) or []),
                video_id=row["video_id"],
            )
            preds.append({
                "sample_id": sid, "stream_order": i,
                "prediction_type": "novel", "virtual_category_id": vid,
            })
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default="pure,ov_assisted")
    ap.add_argument("--subsets", default="full,repeated,balanced")
    ap.add_argument("--methods", default="B0,B1,B2,B3,B4,B5")
    args = ap.parse_args()
    protocols = [p for p in PROTOCOLS if p in args.protocols.split(",")]
    subsets = [s for s in SUBSETS if s in args.subsets.split(",")]
    methods = set(args.methods.split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    orders = stream_orders()
    single_feats = load_mean_features("dinov2", "gt_tracks_single")
    mean_feats = load_mean_features("dinov2", "gt_tracks_mean")
    known_ids = set(json.loads((LEGACY_DATA / "splits" / "known_ids.json").read_text()))
    seed_int = {"main": 1027, "main_seed1027": 1027, "main_seed1028": 1028, "main_seed1029": 1029}

    for proto in protocols:
        gt_rows = load_gt(proto)
        supported = {
            r["ground_truth_category_id"] for r in gt_rows
            if r["protocol_role"] == "supported_known"
        }
        all_rows = []
        for method in ("B0", "B1", "B2", "B3", "B4", "B5"):
            if method not in methods:
                continue
            for subset in subsets:
                for stream in STREAMS:
                    if method == "B0":
                        preds = run_b0_b1(proto, subset, stream, orders[stream], single_feats, gt_rows, supported, seed_int[stream])
                    elif method == "B1":
                        preds = run_b0_b1(proto, subset, stream, orders[stream], mean_feats, gt_rows, supported, seed_int[stream])
                    elif method == "B2":
                        preds = run_b2(subset, stream)
                    elif method == "B3":
                        seed = int(stream.split("_seed")[-1]) if "_seed" in stream else 1027
                        preds = run_b3(subset, stream, seed, orders)
                    elif method == "B4":
                        preds = run_b4(subset, stream)
                    else:
                        preds = run_b5(subset, stream, orders[stream], mean_feats, gt_rows)
                    ev = TrackOCDEvaluator(gt_rows)
                    res = ev.evaluate(preds, subset_ids=subset_ids(proto, subset))
                    row = {
                        "protocol": proto, "method": method, "subset": subset,
                        "seed": stream,
                        **{k: res[k] for k in res if k != "hungarian_assignment"},
                    }
                    all_rows.append(row)
                    (RUNS / f"{proto}_{method}_{subset}_{stream}.json").write_text(
                        json.dumps(row, indent=1, default=str))
                    print(proto, method, subset, stream,
                          "all", round(row["all_track_acc"], 4),
                          "known", round(row["overall_known_acc"], 4),
                          "novel_route", round(row["route_aware_novel_acc"], 4),
                          "novel_cond", round(row["conditional_novel_acc"], 4),
                          "nmi", round(row["novel_only_nmi"], 4),
                          "ari", round(row["novel_only_ari"], 4),
                          flush=True)
        if all_rows:
            path = OUT / f"{proto}_baselines.csv"
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                w.writeheader()
                w.writerows(all_rows)
    print("done")


if __name__ == "__main__":
    main()
