#!/usr/bin/env python3
"""Candidate architecture A: modular Track-then-Discover.

A1: track mean + B2 memory (corrected B2 results, assembled here)
A2: robust medoid / trimmed mean + B2 memory (this script)
A3: track mean + corrected OCD-v2 memory (corrected B4 results)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import hungarian_acc
from src.ocd_v2.common import load_train_known, proxy_split, build_prototypes
from src.ocd_v2.common import load_mean_features
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
from src.trackocd_v1.memory import SeededMultiPrototypeMemory
from src.trackocd_v1.rerun_baselines import (
    STREAMS, load_gt, subset_ids, stream_orders,
)

DATA = PROJECT_ROOT / "data" / "trackocd_v1"
LEGACY = PROJECT_ROOT / "data" / "tao_ow_ocd_v1"
OUT = PROJECT_ROOT / "outputs" / "trackocd_v1" / "metrics"
RUNS = PROJECT_ROOT / "runs" / "trackocd_v1"


def load_frames(encoder, subdir):
    feats = {}
    cache = PROJECT_ROOT / "data" / "caches" / "features" / encoder / subdir
    for p in cache.glob("*.json"):
        r = json.loads(p.read_text())
        feats[r["sample_id"]] = np.asarray(r["frame_embeddings"], dtype=np.float32)
    return feats


def aggregate(frames, method="trimmed"):
    F = np.asarray(frames, dtype=np.float32)
    if len(F) == 1:
        v = F[0]
    elif method == "medoid":
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        sim = Fn @ Fn.T
        v = F[int(np.argmax(sim.mean(axis=1)))]
    else:  # trimmed mean: drop 25% farthest from initial mean
        m = F.mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-12)
        dist = 1.0 - F @ m
        keep = int(np.ceil(len(F) * 0.75))
        idx = np.argsort(dist)[:keep]
        v = F[idx].mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


def simulate_ncm(rows, feats, known_protos, thr):
    novel = {}
    counts = {}
    next_id = 100000
    preds = []
    for i, row in enumerate(rows):
        x = feats[row["sample_id"]]
        best_k, best_s = None, -1.0
        for cid, p in known_protos.items():
            s = float(np.dot(x, p))
            if s > best_s:
                best_k, best_s = cid, s
        if best_s >= thr:
            vid = best_k
        else:
            best_n, best_p = -1.0, None
            for cid, p in novel.items():
                s = float(np.dot(x, p))
                if s > best_n:
                    best_n, best_p = s, cid
            if best_n >= thr:
                vid = best_p
                novel[vid] = (novel[vid] * counts[vid] + x) / (counts[vid] + 1)
                novel[vid] /= np.linalg.norm(novel[vid]) + 1e-12
                counts[vid] += 1
            else:
                vid = next_id
                next_id += 1
                novel[vid] = x.copy()
                counts[vid] = 1
        preds.append({
            "sample_id": row["sample_id"], "stream_order": i,
            "prediction_type": "known" if vid < 100000 else "novel",
            "semantic_category_id": vid if vid < 100000 else None,
            "virtual_category_id": vid if vid >= 100000 else None,
        })
    return preds


def calibrate_thr(train_feats, labels):
    pk, pn = proxy_split(labels, seed=1027)
    ids = sorted(s for s, c in labels.items() if c in pn and s in train_feats)
    X = np.stack([train_feats[s] for s in ids])
    y = np.array([labels[s] for s in ids])
    protos = build_prototypes(train_feats, labels, pk)
    best = (0.45, -1.0)
    for thr in np.arange(0.30, 0.81, 0.025):
        preds = simulate_ncm([{"sample_id": s} for s in ids], train_feats, protos, float(thr))
        pv = np.array([
            (p["semantic_category_id"] if p["prediction_type"] == "known" else 100000 + p["virtual_category_id"])
            for p in preds
        ])
        uniq = sorted(set(int(v) for v in pv))
        remap = {v: i for i, v in enumerate(uniq)}
        pv = np.array([remap[int(v)] for v in pv])
        acc = hungarian_acc(y, pv)[0]
        if acc > best[1]:
            best = (float(thr), acc)
    return best


def main():
    train_dino = load_frames("dinov2", "train_known_mean")
    val_dino = load_frames("dinov2", "gt_tracks_mean")
    _, labels = load_train_known("dinov2")
    orders = stream_orders()
    gt_cache = {proto: load_gt(proto) for proto in ("pure", "ov_assisted")}

    rows = []
    for method in ("medoid", "trimmed"):
        tr_agg = {s: aggregate(f, method) for s, f in train_dino.items()}
        val_agg = {s: aggregate(f, method) for s, f in val_dino.items()}
        thr, proxy_acc = calibrate_thr(tr_agg, labels)
        print(method, "thr", thr, "proxy_acc", round(proxy_acc, 4), flush=True)
        protos = build_prototypes(tr_agg, labels, set(labels.values()))
        for proto in ("pure", "ov_assisted"):
            gt = gt_cache[proto]
            for subset in ("full", "repeated", "balanced"):
                for stream in STREAMS:
                    fname = "val_gt_track_stream.jsonl" if stream == "main" else f"val_gt_track_stream_{stream[5:]}.jsonl"
                    srows = []
                    with open(LEGACY / "public" / fname) as f:
                        for line in f:
                            if line.strip():
                                srows.append(json.loads(line))
                    preds = simulate_ncm(srows, val_agg, protos, thr)
                    ev = TrackOCDEvaluator(gt)
                    res = ev.evaluate(preds, subset_ids=subset_ids(proto, subset))
                    row = {
                        "architecture": f"A2_{method}", "protocol": proto,
                        "subset": subset, "seed": stream,
                        "threshold": thr, "proxy_acc": proxy_acc,
                        **{k: res[k] for k in res if k != "hungarian_assignment"},
                    }
                    rows.append(row)
                    (RUNS / f"A2_{method}_{proto}_{subset}_{stream}.json").write_text(
                        json.dumps(row, indent=1, default=str))
                    print(proto, subset, stream,
                          "all", round(row["all_track_acc"], 4),
                          "known", round(row["overall_known_acc"], 4),
                          "novel_route", round(row["route_aware_novel_acc"], 4),
                          "novel_cond", round(row["conditional_novel_acc"], 4),
                          flush=True)
    # A3: track-mean representation + seeded OCD-v2 memory (controlled)
    val_mean = load_mean_features("dinov2", "gt_tracks_mean")
    tr_mean = load_mean_features("dinov2", "train_known_mean")
    protos = build_prototypes(tr_mean, labels, set(labels.values()))
    mem_params = {
        "attach_thr": 0.525, "create_thr": 0.375, "new_proto_thr": 0.475,
        "max_proto": 4, "ema": 0.25, "maturity_tracks": 2,
    }
    for proto in ("pure", "ov_assisted"):
        gt = gt_cache[proto]
        for subset in ("full", "repeated", "balanced"):
            for stream in STREAMS:
                fname = "val_gt_track_stream.jsonl" if stream == "main" else f"val_gt_track_stream_{stream[5:]}.jsonl"
                srows = []
                with open(LEGACY / "public" / fname) as f:
                    for line in f:
                        if line.strip():
                            srows.append(json.loads(line))
                model = SeededMultiPrototypeMemory(protos, **mem_params)
                preds = []
                for i, row in enumerate(srows):
                    vid = model.predict_one(
                        val_mean[row["sample_id"]], row["sample_id"], i,
                        num_frames=len(row.get("frame_ids", []) or []), video_id=row["video_id"],
                    )
                    preds.append({
                        "sample_id": row["sample_id"], "stream_order": i,
                        "prediction_type": "known" if vid < 200000 else "novel",
                        "semantic_category_id": vid if vid < 200000 else None,
                        "virtual_category_id": vid if vid >= 200000 else None,
                    })
                ev = TrackOCDEvaluator(gt)
                res = ev.evaluate(preds, subset_ids=subset_ids(proto, subset))
                row = {
                    "architecture": "A3_seeded_memory", "protocol": proto,
                    "subset": subset, "seed": stream,
                    **{k: res[k] for k in res if k != "hungarian_assignment"},
                }
                rows.append(row)
                print("A3", proto, subset, stream,
                      "all", round(row["all_track_acc"], 4),
                      "known", round(row["overall_known_acc"], 4),
                      "novel_route", round(row["route_aware_novel_acc"], 4),
                      "novel_cond", round(row["conditional_novel_acc"], 4),
                      flush=True)
    if rows:
        with open(OUT / "modular_architecture.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("done")


if __name__ == "__main__":
    main()
