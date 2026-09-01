#!/usr/bin/env python3
"""Same-support semantic audit: P0 vs P2 on matched novel observations."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1) +
          (bx2 - bx1) * (by2 - by1) - inter)
    return inter / ua if ua > 0 else 0.0


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["image_id"] = int(r["image_id"])
        r["video_id"] = int(r["video_id"])
        r["frame_id"] = int(r["frame_id"])
        r["category_id"] = int(r["category_id"])
        r["gt_category_id"] = int(r["gt_category_id"])
        r["score"] = float(r["score"])
        r["prior_hits"] = int(r["prior_hits"])
        r["bbox"] = json.loads(r["bbox_xyxy"])
    return rows


def match_pairs(a_rows, b_rows):
    by_img = defaultdict(list)
    for r in b_rows:
        by_img[r["image_id"]].append(r)
    pairs = []
    for ra in a_rows:
        best, best_iou, best_rb = None, 0.5, None
        for rb in by_img.get(ra["image_id"], []):
            v = iou(ra["bbox"], rb["bbox"])
            if v > best_iou:
                best_iou, best_rb = v, rb
        if best_rb is not None:
            pairs.append((ra, best_rb))
            by_img[ra["image_id"]].remove(best_rb)
    return pairs


def cls_metrics(rows):
    if not rows:
        return {"n": 0, "clsA": None, "score_mean": None}
    acc = np.mean([int(r["category_id"] == r["gt_category_id"])
                   for r in rows])
    return {
        "n": len(rows),
        "clsA": float(acc),
        "score_mean": float(np.mean([r["score"] for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-prefix", required=True)
    ap.add_argument("--p2-prefix", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    out = {}
    for mode in ("dev", "heldout"):
        a = load(f"{args.p0_prefix}_{mode}.csv")
        b = load(f"{args.p2_prefix}_{mode}.csv")
        a_novel = [r for r in a if r["gt_role"] == "novel"]
        b_novel = [r for r in b if r["gt_role"] == "novel"]
        pairs = match_pairs(a_novel, b_novel)
        same = {
            "n_pairs": len(pairs),
            "clsA_p0": float(np.mean(
                [int(ra["category_id"] == ra["gt_category_id"])
                 for ra, _ in pairs])) if pairs else None,
            "clsA_p2": float(np.mean(
                [int(rb["category_id"] == rb["gt_category_id"])
                 for _, rb in pairs])) if pairs else None,
            "score_mean_p0": float(np.mean(
                [ra["score"] for ra, _ in pairs])) if pairs else None,
            "score_mean_p2": float(np.mean(
                [rb["score"] for _, rb in pairs])) if pairs else None,
            "category_agreement": float(np.mean(
                [int(ra["category_id"] == rb["category_id"])
                 for ra, rb in pairs])) if pairs else None,
            "all_novel_p0": cls_metrics(a_novel),
            "all_novel_p2": cls_metrics(b_novel),
        }
        out[mode] = same
        print(mode, json.dumps(same, indent=2))

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print("SAME_SUPPORT_AUDIT_DONE")


if __name__ == "__main__":
    main()
