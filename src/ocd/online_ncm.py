#!/usr/bin/env python3
"""B2: Online Nearest-Prototype assign-or-create baseline."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_predictions, load_private_labels


def load_mean_features(cache_dir):
    feats = {}
    for p in cache_dir.glob("*.json"):
        r = json.loads(p.read_text())
        feats[r["sample_id"]] = np.asarray(r["mean_embedding"], dtype=np.float32)
    return feats


def load_train_known(encoder):
    cache = PROJECT_ROOT / "data" / "caches" / "features" / encoder / "train_known_mean"
    feats = load_mean_features(cache)
    labels = {}
    with open(PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / "train_known_tracks.jsonl") as f:
        for line in f:
            r = json.loads(line)
            labels[r["sample_id"]] = r["category_id"]
    return feats, labels


def build_prototypes(feats, labels, class_ids):
    protos = {}
    sums = defaultdict(float)
    counts = defaultdict(int)
    for sid, cat in labels.items():
        if cat not in class_ids or sid not in feats:
            continue
        sums[cat] += feats[sid]
        counts[cat] += 1
    for c, s in sums.items():
        v = s / counts[c]
        protos[c] = v / (np.linalg.norm(v) + 1e-12)
    return protos


def simulate(stream_rows, feats, known_protos, threshold):
    novel_protos = {}
    novel_counts = defaultdict(int)
    virtual_to_class = {}
    next_virtual = 100000
    preds = []
    log = []
    for i, row in enumerate(stream_rows):
        sid = row["sample_id"]
        x = feats[sid]
        best_known = -1.0
        best_k = None
        for c, p in known_protos.items():
            s = float(np.dot(x, p))
            if s > best_known:
                best_known, best_k = s, c
        if best_known >= threshold:
            vid = best_k
        else:
            best_n = -1.0
            best_p = None
            for pid, p in novel_protos.items():
                s = float(np.dot(x, p))
                if s > best_n:
                    best_n, best_p = s, pid
            if best_n >= threshold:
                vid = best_p
                novel_protos[vid] = (
                    novel_protos[vid] * novel_counts[vid] + x
                ) / (novel_counts[vid] + 1)
                novel_counts[vid] += 1
                novel_protos[vid] = novel_protos[vid] / (
                    np.linalg.norm(novel_protos[vid]) + 1e-12
                )
            else:
                vid = next_virtual
                next_virtual += 1
                novel_protos[vid] = x
                novel_counts[vid] = 1
        preds.append(vid)
        log.append({"stream_order": i, "sample_id": sid, "virtual_category_id": vid})
    return np.array(preds), log


def assignment_delay(stream_rows, y_true, y_pred, known_mask, pred_map):
    """Delay (in stream units) between first occurrence of a true novel category and
    first assignment to its final matched virtual category."""
    first_true = {}
    first_pred = {}
    for i, (c, p) in enumerate(zip(y_true, y_pred)):
        if known_mask[i]:
            continue
        c = int(c)
        p = int(p)
        first_true.setdefault(c, i)
        if c in pred_map:
            if pred_map[c] == p:
                first_pred.setdefault(c, i)
    return {c: first_pred[c] - first_true[c] for c in first_true if c in first_pred}


def calibrate_threshold(encoder):
    feats, labels = load_train_known(encoder)
    known_classes = sorted(set(labels.values()))
    rng = random.Random(1027)
    rng.shuffle(known_classes)
    n_half = len(known_classes) // 2
    proxy_known = set(known_classes[:n_half])
    proxy_novel = set(known_classes[n_half:])
    pk_protos = build_prototypes(feats, labels, proxy_known)

    proxy_ids = [sid for sid, c in labels.items() if c in proxy_novel and sid in feats]
    proxy_ids.sort()
    proxy_X = np.stack([feats[s] for s in proxy_ids])
    proxy_y = np.array([labels[s] for s in proxy_ids])
    proxy_mask = np.zeros(len(proxy_y), dtype=bool)

    results = {}
    best = None
    for thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        preds, _ = simulate(
            [{"sample_id": s} for s in proxy_ids],
            feats,
            pk_protos,
            thr,
        )
        res = evaluate_predictions(proxy_y, preds, proxy_mask)
        results[thr] = res["acc_all"]
        if best is None or res["acc_all"] > best[1]:
            best = (thr, res["acc_all"])
    return best[0], results, proxy_known, proxy_novel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["dinov2", "clip"], required=True)
    ap.add_argument("--subset", choices=["full", "repeated", "balanced"], default="full")
    args = ap.parse_args()

    threshold, calib_curve, proxy_known, proxy_novel = calibrate_threshold(args.encoder)
    print(f"calibrated threshold: {threshold} curve={calib_curve}")

    feats, labels = load_train_known(args.encoder)
    all_known = set(labels.values())
    known_protos = build_prototypes(feats, labels, all_known)
    val_feats = load_mean_features(
        PROJECT_ROOT / "data" / "caches" / "features" / args.encoder / "gt_tracks_mean"
    )
    private = load_private_labels(PROJECT_ROOT)

    rows_out = []
    manifest_ids = None
    if args.subset != "full":
        manifest_ids = set(
            json.loads(
                (
                    PROJECT_ROOT
                    / "data"
                    / "tao_ow_ocd_v1"
                    / "manifests"
                    / f"{args.subset}_track_ids.json"
                ).read_text()
            )
        )

    for stream_name in ["val_gt_track_stream.jsonl", "val_gt_track_stream_seed1027.jsonl",
                        "val_gt_track_stream_seed1028.jsonl", "val_gt_track_stream_seed1029.jsonl"]:
        rows = []
        with open(PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / stream_name) as f:
            for line in f:
                r = json.loads(line)
                if r["sample_id"] in val_feats:
                    rows.append(r)
        preds, log = simulate(rows, val_feats, known_protos, threshold)
        y_true = np.array([private[r["sample_id"]]["ground_truth_category_id"] for r in rows])
        known_mask = np.array([private[r["sample_id"]]["is_known"] for r in rows])
        if manifest_ids is not None:
            keep = np.array([r["sample_id"] in manifest_ids for r in rows])
            y_true_e, preds_e, known_e = y_true[keep], preds[keep], known_mask[keep]
        else:
            y_true_e, preds_e, known_e = y_true, preds, known_mask
        res = evaluate_predictions(y_true_e, preds_e, known_e)

        # delay
        w = np.zeros((int(preds.max()) + 1, int(y_true.max()) + 1), dtype=int)
        np.add.at(w, (preds, y_true), 1)
        r_, c_ = linear_sum_assignment(w.max() - w)
        pred_map = {int(c): int(p) for p, c in zip(r_, c_)}
        delays = assignment_delay(rows, y_true, preds, known_mask, pred_map)
        res["assignment_delay"] = delays
        res["mean_assignment_delay"] = float(np.mean(list(delays.values()))) if delays else None
        res["seed"] = stream_name.replace("val_gt_track_stream", "main").replace(".jsonl", "").replace("_seed", "_seed")
        res["encoder"] = args.encoder
        res["subset"] = args.subset
        res["threshold"] = threshold
        res["calibration_curve"] = calib_curve
        res["proxy_known_classes"] = sorted(proxy_known)
        res["proxy_novel_classes"] = sorted(proxy_novel)
        res["prediction_log"] = log
        rows_out.append(res)

        out_dir = PROJECT_ROOT / "runs" / "gt_online_ncm"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{args.encoder}_{args.subset}_{res['seed']}.json").write_text(
            json.dumps(res, indent=1, default=str)
        )
        print(json.dumps(res, indent=1, default=str))

    out_csv = PROJECT_ROOT / "outputs" / "metrics" / "gt_online_ncm.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "a") as f:
        for res in rows_out:
            f.write(
                f"ncm,{res['encoder']},{res['subset']},{res['seed']},{threshold},"
                f"{res['acc_all']:.4f},{res['acc_known']:.4f},{res['acc_novel']:.4f},"
                f"{res['nmi']:.4f},{res['ari']:.4f},{res['predicted_categories']},"
                f"{res['category_count_abs_error']},{res['mean_fragmentation']:.4f},"
                f"{res['merge_error']:.4f},{res['duplicate_creation_rate']:.4f}\n"
            )


if __name__ == "__main__":
    main()
