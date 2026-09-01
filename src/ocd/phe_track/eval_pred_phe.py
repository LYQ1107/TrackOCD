#!/usr/bin/env python3
"""B4: PHE-Track on SimOWT predicted tracks matched to GT tracks."""
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
from src.evaluation.track_matching import load_gt_tracks, load_pred_tracks, match_tracks
from src.ocd.phe_track.phe_track_model import PPNetTrack
from src.ocd.phe_track.eval_phe_track import (
    assignment_delay,
    calibrate_radius,
    class_hash_center,
    load_mean_features,
    load_train_known,
    simulate,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["dinov2", "clip"], required=True)
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--iou", type=float, default=0.5)
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
    radius, calib = calibrate_radius(model, feats, labels, proxy_known, proxy_novel, class_ids)
    print(f"radius={radius} calib={calib}", flush=True)
    cid2idx = {c: i for i, c in enumerate(class_ids)}
    all_known_centers = [(cid2idx[c], class_hash_center(model, cid2idx[c])) for c in class_ids]

    gt_vid, gt_anns = load_gt_tracks()
    pred_vid, pred_anns, pred_rows = load_pred_tracks()
    matches = match_tracks(gt_anns, pred_anns, args.iou)
    gt_sample = {(vid, tid): rec["sample_id"] for vid, td in gt_vid.items() for tid, rec in td.items()}
    pred_sample = {(r["video_id"], r["track_id"]): r["sample_id"] for r in pred_rows}
    private = load_private_labels(PROJECT_ROOT)
    matches = [m for m in matches if gt_sample[(m[0], m[1])] in private]
    matched_pred_ids = {pred_sample[(vid, p)] for vid, g, p, iou in matches}
    matched_gt_ids = {gt_sample[(vid, g)] for vid, g, p, iou in matches}

    pred_feats = load_mean_features(
        PROJECT_ROOT / "data" / "caches" / "features" / args.encoder / "pred_tracks_mean"
    )
    rows = [r for r in pred_rows if r["sample_id"] in matched_pred_ids and r["sample_id"] in pred_feats]
    rows.sort(key=lambda r: (r["video_id"], r["stream_order"]))
    for i, r in enumerate(rows):
        r["stream_order"] = i

    preds, log = simulate(model, rows, pred_feats, all_known_centers, radius)
    gt_sid_by_pred = {
        pred_sample[(vid, p)]: gt_sample[(vid, g)]
        for vid, g, p, iou in matches
    }
    y_true = np.array([private[gt_sid_by_pred[r["sample_id"]]]["ground_truth_category_id"] for r in rows])
    known_mask = np.array([private[gt_sid_by_pred[r["sample_id"]]]["is_known"] for r in rows])
    res = evaluate_predictions(y_true, preds, known_mask)

    w = np.zeros((int(preds.max()) + 1, int(y_true.max()) + 1), dtype=int)
    np.add.at(w, (preds, y_true), 1)
    r_, c_ = linear_sum_assignment(w.max() - w)
    pred_map = {int(c): int(p) for p, c in zip(r_, c_)}
    delays = assignment_delay(rows, y_true, preds, known_mask, pred_map)
    res["assignment_delay"] = delays
    res["mean_assignment_delay"] = float(np.mean(list(delays.values()))) if delays else None
    res["encoder"] = args.encoder
    res["seed"] = args.seed
    res["iou_threshold"] = args.iou
    res["radius"] = radius
    res["radius_calibration"] = calib
    res["num_pred_tracks_evaluated"] = len(rows)
    res["num_matched_pairs"] = len(matches)
    res["matched_gt_ids"] = sorted(matched_gt_ids)
    res["matched_pred_ids"] = sorted(matched_pred_ids)
    res["prediction_log"] = log

    out_dir = PROJECT_ROOT / "runs" / "pred_phe_track"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.encoder}_seed{args.seed}_iou{args.iou}.json").write_text(
        json.dumps(res, indent=1, default=str)
    )
    print(json.dumps(res, indent=1, default=str))

    out_csv = PROJECT_ROOT / "outputs" / "metrics" / "pred_phe_track.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "a") as f:
        f.write(
            f"phe_track_pred,{res['encoder']},{res['seed']},{args.iou},{radius},"
            f"{res['acc_all']:.4f},{res['acc_known']:.4f},{res['acc_novel']:.4f},"
            f"{res['nmi']:.4f},{res['ari']:.4f},{res['predicted_categories']},"
            f"{res['category_count_abs_error']},{res['mean_fragmentation']:.4f},"
            f"{res['merge_error']:.4f},{res['duplicate_creation_rate']:.4f}\n"
        )


if __name__ == "__main__":
    main()
