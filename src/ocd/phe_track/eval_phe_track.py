#!/usr/bin/env python3
"""B3: run trained PHE-Track strictly online over the TAO-OW GT track stream."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_predictions, load_private_labels
from src.ocd.phe_track.phe_track_model import PPNetTrack


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
            if r["sample_id"] in feats:
                labels[r["sample_id"]] = r["category_id"]
    return feats, labels


def hash_code(model, x):
    model.eval()
    with torch.no_grad():
        h = model(torch.from_numpy(x).unsqueeze(0).cuda()).cpu()
    return (h[0] > 0).int().cpu().numpy()


def class_hash_center(model, class_idx):
    with torch.no_grad():
        proto = model.prototype_vectors_global[
            class_idx * model.global_proto_per_class : (class_idx + 1) * model.global_proto_per_class
        ].mean(0)
        h = torch.tanh(model.hash_head(proto.unsqueeze(0)) * 3).sign()
    return (h[0] > 0).int().cpu().numpy()


def simulate(model, rows, feats, known_centers, radius):
    novel_centers = []
    preds = []
    log = []
    for i, row in enumerate(rows):
        code = hash_code(model, feats[row["sample_id"]])
        best = radius + 1
        best_id = None
        for cid, c in known_centers:
            d = int(np.count_nonzero(code != c))
            if d < best:
                best, best_id = d, cid
        if best_id is None:
            for cid, c in novel_centers:
                d = int(np.count_nonzero(code != c))
                if d < best:
                    best, best_id = d, cid
        if best_id is None or best > radius:
            cid = 100000 + len(novel_centers)
            novel_centers.append((cid, code))
            best_id = cid
        preds.append(best_id)
        log.append({"stream_order": i, "sample_id": row["sample_id"], "virtual_category_id": best_id})
    return np.array(preds), log


def assignment_delay(rows, y_true, y_pred, known_mask, pred_map):
    first_true = {}
    first_pred = {}
    for i, (c, p) in enumerate(zip(y_true, y_pred)):
        if known_mask[i]:
            continue
        c, p = int(c), int(p)
        first_true.setdefault(c, i)
        if pred_map.get(c) == p:
            first_pred.setdefault(c, i)
    return {c: first_pred[c] - first_true[c] for c in first_true if c in first_pred}


def calibrate_radius(model, feats, labels, proxy_known, proxy_novel, class_ids):
    cid2idx = {c: i for i, c in enumerate(class_ids)}
    proxy_ids = sorted(s for s, c in labels.items() if c in proxy_known or c in proxy_novel)
    rows = [{"sample_id": s} for s in proxy_ids]
    y = np.array([cid2idx[labels[s]] for s in proxy_ids])
    known_mask = np.array([labels[s] in proxy_known for s in proxy_ids])
    known_centers = [(cid2idx[c], class_hash_center(model, cid2idx[c])) for c in sorted(proxy_known)]
    results = {}
    for radius in [1, 2, 3]:
        preds, _ = simulate(model, rows, feats, known_centers, radius)
        res = evaluate_predictions(y, preds, known_mask)
        results[radius] = {
            "acc_all": res["acc_all"],
            "acc_known": res["acc_known"],
            "acc_novel": res["acc_novel"],
            "score": 0.5 * (res["acc_known"] + res["acc_novel"]),
        }
    best = max(results, key=lambda r: results[r]["score"])
    return best, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["dinov2", "clip"], required=True)
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--subset", choices=["full", "repeated", "balanced"], default="full")
    ap.add_argument("--radius", type=int, default=None, help="fixed Hamming radius (skip calibration)")
    ap.add_argument("--eval-ids-file", type=str, default=None, help="only evaluate these sample ids")
    args = ap.parse_args()

    ckpt_dir = PROJECT_ROOT / "runs" / "phe_track" / f"{args.encoder}_seed{args.seed}"
    ckpt = torch.load(ckpt_dir / "checkpoint.pth", map_location="cpu")
    class_ids = ckpt["class_ids"]
    model = PPNetTrack(
        in_dim=768 if args.encoder == "dinov2" else 512,
        prototype_dim=768 if args.encoder == "dinov2" else 512,
        num_classes=len(class_ids),
        global_proto_per_class=10,
        hash_code_length=12,
    )
    model.load_state_dict(ckpt["ema"])
    model.cuda().eval()

    feats, labels = load_train_known(args.encoder)
    rng = random.Random(1027)
    known_classes = sorted(set(labels.values()))
    rng.shuffle(known_classes)
    n_half = len(known_classes) // 2
    proxy_known = set(known_classes[:n_half])
    proxy_novel = set(known_classes[n_half:])
    if args.radius is not None:
        radius, calib = args.radius, {"fixed": args.radius}
    else:
        radius, calib = calibrate_radius(model, feats, labels, proxy_known, proxy_novel, class_ids)
    print(f"radius={radius} calib={calib}", flush=True)

    cid2idx = {c: i for i, c in enumerate(class_ids)}
    all_known_centers = [
        (cid2idx[c], class_hash_center(model, cid2idx[c])) for c in class_ids
    ]

    val_feats = load_mean_features(
        PROJECT_ROOT / "data" / "caches" / "features" / args.encoder / "gt_tracks_mean"
    )
    private = load_private_labels(PROJECT_ROOT)
    manifest_ids = None
    if args.subset != "full":
        manifest_ids = set(
            json.loads(
                (
                    PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "manifests" / f"{args.subset}_track_ids.json"
                ).read_text()
            )
        )

    rows_out = []
    for stream_name in [
        "val_gt_track_stream.jsonl",
        "val_gt_track_stream_seed1027.jsonl",
        "val_gt_track_stream_seed1028.jsonl",
        "val_gt_track_stream_seed1029.jsonl",
    ]:
        rows = []
        with open(PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / stream_name) as f:
            for line in f:
                r = json.loads(line)
                if r["sample_id"] in val_feats:
                    rows.append(r)
        preds, log = simulate(model, rows, val_feats, all_known_centers, radius)
        y_true = np.array([private[r["sample_id"]]["ground_truth_category_id"] for r in rows])
        known_mask = np.array([private[r["sample_id"]]["is_known"] for r in rows])
        if args.eval_ids_file is not None:
            eval_ids = set(json.loads((PROJECT_ROOT / args.eval_ids_file).read_text()))
            keep = np.array([r["sample_id"] in eval_ids for r in rows])
            y_true_e, preds_e, known_e = y_true[keep], preds[keep], known_mask[keep]
            res = evaluate_predictions(y_true_e, preds_e, known_e)
        else:
            res = evaluate_predictions(y_true, preds, known_mask)
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
        res["radius"] = radius
        res["radius_calibration"] = calib
        res["prediction_log"] = log
        rows_out.append(res)

        out_dir = PROJECT_ROOT / "runs" / "gt_phe_track"
        out_dir.mkdir(parents=True, exist_ok=True)
        radius_suffix = f"_r{radius}" if args.radius is not None else ""
        if args.eval_ids_file is not None:
            radius_suffix += "_matched"
        (out_dir / f"{args.encoder}_seed{args.seed}_{args.subset}_{res['seed']}{radius_suffix}.json").write_text(
            json.dumps(res, indent=1, default=str)
        )
        print(json.dumps(res, indent=1, default=str))

    out_csv = PROJECT_ROOT / "outputs" / "metrics" / "gt_phe_track.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "a") as f:
        for res in rows_out:
            f.write(
                f"phe_track,{res['encoder']},{res['seed']},{res['subset']},{radius},"
                f"{res['acc_all']:.4f},{res['acc_known']:.4f},{res['acc_novel']:.4f},"
                f"{res['nmi']:.4f},{res['ari']:.4f},{res['predicted_categories']},"
                f"{res['category_count_abs_error']},{res['mean_fragmentation']:.4f},"
                f"{res['merge_error']:.4f},{res['duplicate_creation_rate']:.4f}\n"
            )


if __name__ == "__main__":
    main()
