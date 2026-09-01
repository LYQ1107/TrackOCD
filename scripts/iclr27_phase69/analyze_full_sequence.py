#!/usr/bin/env python3
"""Analyze a Phase69 OVTR prediction stream against validation annotations.

The model never receives annotations here; GT is consumed only by this
post-hoc evaluator.  Metrics mirror the Phase68 full-sequence diagnostic and
retain exact row denominators.  TrackEval is run separately on the same JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import tempfile
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_GT = ROOT / "data/external_annotations/ovtr/validation_ours_v1.json"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def xywh_to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in box[:4])
    return x, y, x + max(w, 0.0), y + max(h, 0.0)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + ab - inter
    return inter / den if den > 0 else 0.0


def load_records(gt_path: pathlib.Path, pred_path: pathlib.Path):
    gt_doc = json.loads(gt_path.read_text())
    pred_doc = json.loads(pred_path.read_text())
    gt: Dict[int, List[dict]] = defaultdict(list)
    for ann in gt_doc.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        gt[int(ann["image_id"])].append(ann)
    pred: Dict[int, List[dict]] = defaultdict(list)
    for row in pred_doc:
        if "image_id" in row and "bbox" in row:
            pred[int(row["image_id"])].append(row)
    images = {int(x["id"]): x for x in gt_doc.get("images", [])}
    return gt, pred, images, len(pred_doc)


def recall_curve(gt: Dict[int, List[dict]], pred: Dict[int, List[dict]],
                 topks: Iterable[int], thresholds: Iterable[float]) -> Dict[str, Any]:
    totals = sum(len(v) for v in gt.values())
    out: Dict[str, Any] = {"gt_rows": totals, "topk": {}}
    for k in topks:
        hits = {float(t): 0 for t in thresholds}
        best_values: List[float] = []
        images_with_prediction = 0
        for image_id, anns in gt.items():
            ps = sorted(pred.get(image_id, []), key=lambda x: float(x.get("score", 0.0)), reverse=True)
            if ps:
                images_with_prediction += 1
            if k > 0:
                ps = ps[:k]
            for ann in anns:
                best = max((iou(ann["bbox"], p["bbox"]) for p in ps), default=0.0)
                best_values.append(best)
                for t in hits:
                    if best >= t:
                        hits[t] += 1
        den = max(totals, 1)
        out["topk"][str(k)] = {
            "thresholds": {f"{t:.1f}": {"matched_rows": n, "recall": n / den}
                           for t, n in sorted(hits.items())},
            "mean_best_iou": statistics.fmean(best_values) if best_values else 0.0,
            "median_best_iou": statistics.median(best_values) if best_values else 0.0,
            "images_with_gt": sum(bool(v) for v in gt.values()),
            "images_with_prediction": images_with_prediction,
        }
    return out


def continuity_proxy(gt: Dict[int, List[dict]], pred: Dict[int, List[dict]], images: Dict[int, dict]) -> Dict[str, Any]:
    by_track: Dict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)
    for image_id, anns in gt.items():
        frame = int(images.get(image_id, {}).get("frame_index", images.get(image_id, {}).get("frame_id", 0)))
        ps = pred.get(image_id, [])
        for ann in anns:
            best = max((iou(ann["bbox"], p["bbox"]) for p in ps), default=0.0)
            vid = int(images.get(image_id, {}).get("video_id", ann.get("video_id", -1)))
            tid = int(ann.get("track_id", ann.get("instance_id", -1)))
            by_track[(vid, tid)].append((frame, best))
    details = []
    for (vid, tid), values in by_track.items():
        values.sort()
        hits = [int(v >= 0.5) for _, v in values]
        longest = cur = 0
        for hit in hits:
            cur = cur + 1 if hit else 0
            longest = max(longest, cur)
        details.append({"video_id": vid, "track_id": tid, "frames": len(values),
                        "reliable_frames": sum(hits),
                        "reliable_fraction": sum(hits) / max(len(hits), 1),
                        "longest_reliable_run": longest})
    return {
        "gt_tracks": len(details),
        "mean_reliable_fraction": statistics.fmean(x["reliable_fraction"] for x in details) if details else 0.0,
        "median_reliable_fraction": statistics.median(x["reliable_fraction"] for x in details) if details else 0.0,
        "mean_longest_reliable_run": statistics.fmean(x["longest_reliable_run"] for x in details) if details else 0.0,
        "tracks_reliable_at_least_once": sum(x["reliable_frames"] > 0 for x in details),
        "details": details,
        "definition": "diagnostic class-agnostic IoU>=0.5 frame proxy; not a TrackEval score",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-json", type=pathlib.Path, required=True)
    ap.add_argument("--gt-json", type=pathlib.Path, default=DEFAULT_GT)
    ap.add_argument("--out-json", type=pathlib.Path, required=True)
    args = ap.parse_args()
    gt, pred, images, n_predictions = load_records(args.gt_json, args.pred_json)
    obj = {
        "protocol": "trackocd_phase69_ovtr_full_sequence_validation",
        "prediction": {"path": str(args.pred_json.resolve()), "bytes": args.pred_json.stat().st_size,
                       "sha256": sha256(args.pred_json), "count": n_predictions,
                       "images": len(pred)},
        "gt": {"path": str(args.gt_json.resolve()), "bytes": args.gt_json.stat().st_size,
               "sha256": sha256(args.gt_json), "images": len(gt),
               "annotations": sum(len(v) for v in gt.values())},
        "recall": recall_curve(gt, pred, (1, 5, 20, 100, 0), (0.3, 0.5, 0.7)),
        "track_continuity_proxy": continuity_proxy(gt, pred, images),
        "labels_used_for_model": False,
        "sealed_public_q1_accessed": False,
        "held_event_gt_used_for_model": False,
    }
    atomic_json(args.out_json, obj)
    print(json.dumps({"predictions": n_predictions,
                      "gt_rows": obj["gt"]["annotations"],
                      "recall_top20_iou05": obj["recall"]["topk"]["20"]["thresholds"]["0.5"]["recall"],
                      "out": str(args.out_json)}, indent=2))


if __name__ == "__main__":
    main()
