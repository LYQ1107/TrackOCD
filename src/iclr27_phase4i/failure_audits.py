"""Tracking-native failure, FP semantic pollution, and fragmentation vs
semantic continuity audits on the 20-video subset.

Inputs:
  - B0 predictions (original association),
  - semantic logs (B1 or B2),
  - TAO GT (offline evaluation only).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TAO_JSON = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset" / "validation_20.json"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt():
    d = json.loads(TAO_JSON.read_text())
    img_by_id = {im["id"]: im for im in d["images"]}
    out = defaultdict(list)
    for ann in d["annotations"]:
        b = ann["bbox"]
        out[ann["image_id"]].append({
            "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            "track_id": int(ann["track_id"]),
            "category_id": int(ann["category_id"]),
            "video_id": int(ann["video_id"]),
            "frame_index": img_by_id[ann["image_id"]]["frame_index"],
        })
    return out


def load_preds(pred_dir):
    out = defaultdict(list)
    for p in pred_dir.glob("*.json"):
        image_id = int(p.stem)
        for r in json.loads(p.read_text()):
            b = r["bbox"]
            out[image_id].append({
                "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
                "track_id": int(r["track_id"]),
                "score": float(r["score"]),
            })
    return out


def load_semantic_logs(log_root):
    out = {}
    for p in sorted(log_root.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                key = (int(r["image_id"]),
                       tuple(round(v, 1) for v in r["bbox"]))
                out[key] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--sem-log-root", type=Path, default=None)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()

    gt = load_gt()
    preds = load_preds(args.pred_dir)
    sem = load_semantic_logs(args.sem_log_root) if args.sem_log_root else {}

    # per-prediction GT association
    pred_gt = {}
    for image_id, plist in preds.items():
        for i, p in enumerate(plist):
            best, bi = None, 0.5
            for g in gt.get(image_id, []):
                v = iou(p["bbox"], g["bbox"])
                if v >= bi:
                    bi, best = v, g
            pred_gt[(image_id, i)] = best

    # tracklet stats from B0 predictions
    tracks = defaultdict(list)
    for image_id in sorted(preds):
        for i, p in enumerate(preds[image_id]):
            g = pred_gt.get((image_id, i))
            tracks[p["track_id"]].append({
                "image_id": image_id, "i": i, "score": p["score"],
                "bbox": p["bbox"], "gt": g,
            })
    rows = []
    novel_birth_fp = 0
    novel_birth_total = 0
    fp_sem_rows = []
    for tid, dets in tracks.items():
        dets.sort(key=lambda d: d["image_id"])
        gaps = []
        for a, b in zip(dets, dets[1:]):
            gaps.append(int(b["image_id"]) - int(a["image_id"]))
        max_gap = max(gaps) if gaps else 0
        matched = sum(1 for d in dets if d["gt"] is not None)
        fp = len(dets) - matched
        rows.append({
            "track_id": tid, "length": len(dets), "matched": matched,
            "fp": fp, "max_gap": max_gap, "mean_score": float(
                np.mean([d["score"] for d in dets])),
            "gt_video": dets[0]["gt"]["video_id"] if dets[0]["gt"] else "",
            "gt_track": dets[0]["gt"]["track_id"] if dets[0]["gt"] else "",
        })
        # FP semantic pollution: detections without GT, semantic action
        sem_by_img = defaultdict(list)
        for (img, _b), r in sem.items():
            sem_by_img[img].append(r)
        for d in dets:
            if d["gt"] is not None:
                continue
            best, bi = None, 0.9
            for r in sem_by_img.get(d["image_id"], []):
                v = iou(r["bbox"], d["bbox"])
                if v >= bi:
                    bi, best = v, r
            if best is None:
                continue
            novel_birth_total += 1
            if best["semantic_action"] == "NOVEL":
                novel_birth_fp += 1
                fp_sem_rows.append(best)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "tracklets": len(rows),
        "fp_tracklets_share": np.mean([r["fp"] > 0 for r in rows]),
        "mean_tracklet_length": float(np.mean([r["length"] for r in rows])),
        "fragmented_gt_share": None,
        "fp_semantic_novel_rate": novel_birth_fp /
            max(novel_birth_total, 1),
        "fp_semantic_rows": novel_birth_fp,
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
