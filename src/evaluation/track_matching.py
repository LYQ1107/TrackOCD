"""Hungarian matching between predicted tracks and GT tracks using track-level
temporal IoU (bbox/frame overlap only, no category information)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def temporal_iou(gt_anns, pred_anns):
    """gt_anns/pred_anns: dict frame_id -> box_xyxy."""
    gt_frames = set(gt_anns.keys())
    pred_frames = set(pred_anns.keys())
    common = gt_frames & pred_frames
    if not common:
        return 0.0
    iou_sum = sum(box_iou(gt_anns[f], pred_anns[f]) for f in common)
    return iou_sum / len(gt_frames | pred_frames)


def match_tracks(gt_by_video, pred_by_video, threshold=0.5):
    """Returns per-video matched list of (gt_sample, pred_sample, iou)."""
    matches = []
    for vid in sorted(set(list(gt_by_video.keys()) + list(pred_by_video.keys()))):
        gts = gt_by_video.get(vid, {})
        preds = pred_by_video.get(vid, {})
        gt_ids = sorted(gts.keys())
        pred_ids = sorted(preds.keys())
        if not gt_ids or not pred_ids:
            continue
        S = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
        for i, g in enumerate(gt_ids):
            for j, p in enumerate(pred_ids):
                S[i, j] = temporal_iou(gts[g], preds[p])
        rows, cols = linear_sum_assignment(-S)
        for r, c in zip(rows, cols):
            iou = S[r, c]
            if iou >= threshold:
                matches.append((vid, gt_ids[r], pred_ids[c], float(iou)))
    return matches


def load_gt_tracks(project_root=PROJECT_ROOT):
    gt_data = json.load(
        open(project_root / "data" / "raw" / "tao" / "annotations" / "validation.json")
    )
    img_to_frame = {im["id"]: im["frame_index"] for im in gt_data["images"]}
    img_to_name = {im["id"]: im["file_name"] for im in gt_data["images"]}
    by_video = defaultdict(dict)
    by_video_anns = defaultdict(dict)
    for ann in gt_data["annotations"]:
        vid, tid = ann["video_id"], ann["track_id"]
        x, y, w, h = ann["bbox"]
        key = (vid, tid)
        if key not in by_video[vid]:
            by_video[vid][tid] = {
                "sample_id": f"{vid}_{tid}",
                "video_id": vid,
                "track_id": tid,
                "category_id": ann["category_id"],
                "frame_ids": [],
                "image_paths": [],
                "boxes_xyxy": [],
            }
        rec = by_video[vid][tid]
        rec["frame_ids"].append(ann["image_id"])
        rec["image_paths"].append(img_to_name[ann["image_id"]])
        rec["boxes_xyxy"].append([x, y, x + w, y + h])
        by_video_anns[vid][tid] = by_video_anns[vid].get(tid, {})
        by_video_anns[vid][tid][ann["image_id"]] = [x, y, x + w, y + h]
    for vid in by_video:
        for tid in by_video[vid]:
            order = np.argsort([img_to_frame[f] for f in by_video[vid][tid]["frame_ids"]])
            by_video[vid][tid]["frame_ids"] = [by_video[vid][tid]["frame_ids"][i] for i in order]
            by_video[vid][tid]["image_paths"] = [by_video[vid][tid]["image_paths"][i] for i in order]
            by_video[vid][tid]["boxes_xyxy"] = [by_video[vid][tid]["boxes_xyxy"][i] for i in order]
    return by_video, by_video_anns


def load_pred_tracks(project_root=PROJECT_ROOT):
    rows = []
    with open(project_root / "data" / "tao_ow_ocd_v1" / "public" / "pred_track_stream.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    by_video = defaultdict(dict)
    by_video_anns = defaultdict(dict)
    for r in rows:
        vid, tid = r["video_id"], r["track_id"]
        by_video[vid][tid] = r
        anns = {}
        for fid, box in zip(r["frame_ids"], r["boxes_xyxy"]):
            anns[fid] = box
        by_video_anns[vid][tid] = anns
    return by_video, by_video_anns, rows


def main():
    gt_vid, gt_anns = load_gt_tracks()
    pred_vid, pred_anns, pred_rows = load_pred_tracks()
    private = {}
    with open(PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "private" / "val_gt_track_labels.jsonl") as f:
        for line in f:
            r = json.loads(line)
            private[r["sample_id"]] = r
    for threshold in (0.5, 0.3):
        matches = match_tracks(gt_anns, pred_anns, threshold)
        gt_sample = {(vid, tid): rec["sample_id"] for vid, td in gt_vid.items() for tid, rec in td.items()}
        pred_sample = {(r["video_id"], r["track_id"]): r["sample_id"] for r in pred_rows}
        # only non-distractor GT tracks are in the private label set
        matches = [m for m in matches if gt_sample[(m[0], m[1])] in private]
        matched_gt = {gt_sample[(vid, g)] for vid, g, p, iou in matches}
        matched_pred = {pred_sample[(vid, p)] for vid, g, p, iou in matches}
        all_gt = {sid for sid in gt_sample.values() if sid in private}
        all_pred = set(pred_sample.values())
        gt_known = sum(1 for m in matched_gt if private[m]["is_known"])
        gt_unknown = sum(1 for m in matched_gt if not private[m]["is_known"])
        all_known = sum(1 for m in all_gt if private[m]["is_known"])
        all_unknown = sum(1 for m in all_gt if not private[m]["is_known"])
        out = {
            "threshold": threshold,
            "matched": len(matches),
            "unmatched_gt": len(all_gt - matched_gt),
            "unmatched_pred": len(all_pred - matched_pred),
            "gt_track_coverage": len(matched_gt) / len(all_gt),
            "pred_track_precision": len(matched_pred) / len(all_pred),
            "known_coverage": gt_known / all_known if all_known else 0.0,
            "unknown_coverage": gt_unknown / all_unknown if all_unknown else 0.0,
            "matched_known": gt_known,
            "matched_unknown": gt_unknown,
            "matched_categories": sorted(
                {private[gt_sample[(vid, g)]]["ground_truth_category_id"] for vid, g, p, iou in matches}
            ),
        }
        out_path = PROJECT_ROOT / "outputs" / "metrics" / f"track_matching_iou{threshold}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        if threshold == 0.5:
            (PROJECT_ROOT / "outputs" / "metrics" / "matched_gt_ids_iou0.5.json").write_text(
                json.dumps(sorted(matched_gt))
            )
            matched_rows = [
                r
                for r in pred_rows
                if r["sample_id"] in matched_pred
            ]
            matched_rows.sort(key=lambda r: (r["video_id"], r["stream_order"]))
            for i, r in enumerate(matched_rows):
                r["stream_order"] = i
            out_stream = PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / "pred_track_stream_matched_iou0.5.jsonl"
            with open(out_stream, "w") as f:
                for r in matched_rows:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
            print(f"matched pred stream: {len(matched_rows)} tracks -> {out_stream}")


if __name__ == "__main__":
    main()
