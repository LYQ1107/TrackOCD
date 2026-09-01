#!/usr/bin/env python3
"""Build COVTrack proposal populations for dev/heldout with strict
(video_id, frame_id, proposal_local_id) keys and GT role labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
RESULTS = ROOT / "runs" / "iclr27_phase4p" / "covtrack_eval" / "results" / "tao_track.json"
TAO_VAL = ROOT / "data" / "raw" / "tao" / "annotations" / "validation.json"
DEV_GT = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset" / "validation_20.json"
HO_GT = ROOT / "outputs" / "iclr27_phase4n" / "audit" / "validation_heldout_tao_corrected.json"
OUT = ROOT / "outputs" / "iclr27_phase4p" / "covtrack_tco"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / ua if ua > 0 else 0.0


def load_gt(path, known_ids):
    d = json.loads(path.read_text())
    gt = {}
    for a in d["annotations"]:
        if a.get("iscrowd"):
            continue
        b = a["bbox"]
        gt.setdefault(int(a["image_id"]), []).append(
            {
                "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
                "role": "known" if int(a["category_id"]) in known_ids else "novel",
            }
        )
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "heldout"], required=True)
    args = ap.parse_args()

    known_ids = set(json.loads(
        (ROOT / "data" / "trackocd_v1" / "pure" / "splits" / "supported_known_ids.json").read_text()
    ))
    gt_path = DEV_GT if args.mode == "dev" else HO_GT
    gt = load_gt(gt_path, known_ids)
    gt_images = {im["id"] for im in json.loads(gt_path.read_text())["images"]}

    val = json.loads(TAO_VAL.read_text())
    img_map = {im["id"]: im for im in val["images"]}

    rows = json.loads(RESULTS.read_text())
    # filter to split
    kept = [r for r in rows if int(r["image_id"]) in gt_images]
    # group per frame for local proposal id
    per_frame = {}
    for r in kept:
        per_frame.setdefault(int(r["image_id"]), []).append(r)
    for img_id in per_frame:
        per_frame[img_id].sort(key=lambda r: (int(r.get("track_id", -1)), float(r.get("score", 0))))
        for i, r in enumerate(per_frame[img_id]):
            r["_proposal_local_id"] = i

    out_rows = []
    n_known = n_novel = n_fp = 0
    for r in kept:
        img_id = int(r["image_id"])
        im = img_map[img_id]
        bb = [float(v) for v in r["bbox"][:4]]
        bbox = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
        best, role = 0.5, "fp"
        for g in gt.get(img_id, []):
            v = iou(bbox, g["bbox"])
            if v >= best:
                best, role = v, g["role"]
        out_rows.append(
            {
                "video_id": int(r["video_id"]),
                "frame_id": int(im["frame_index"]),
                "image_id": img_id,
                "proposal_local_id": r["_proposal_local_id"],
                "track_id": int(r.get("track_id", -1)),
                "score": float(r["score"]),
                "bbox_xyxy": json.dumps(bbox),
                "bbox_area": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                "bbox_aspect": float((bbox[2] - bbox[0]) / max(1e-6, bbox[3] - bbox[1])),
                "category_id": int(r.get("category_id", -1)),
                "gt_role": role,
                "gt_iou": float(best),
            }
        )
        if role == "known":
            n_known += 1
        elif role == "novel":
            n_novel += 1
        else:
            n_fp += 1

    out_rows.sort(key=lambda r: (r["video_id"], r["frame_id"], r["proposal_local_id"]))
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"proposal_population_{args.mode}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"{args.mode}: proposals={len(out_rows)} known={n_known} novel={n_novel} fp={n_fp} -> {out_csv}")


if __name__ == "__main__":
    main()
