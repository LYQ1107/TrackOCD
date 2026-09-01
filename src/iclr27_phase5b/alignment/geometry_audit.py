"""Category-agnostic vs class-aware physical alignment audit (Phase 5B).

For every physical track in the Phase 5A Q1 stream:
  - frame-level max IoU against every GT box on the same image;
  - track-level best GT track (temporal mean IoU, greedy one-to-one);
  - many-to-one / one-to-many overlaps;
  - fragmentation and duplicate-active-track indicators;
  - predicted category distribution and category-match vs GT.

This is a diagnostic only: it never selects a model or changes the stream.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.protocol import (
    Q1_DEV,
    group_tracks,
    known_ids,
    load_gt_tracks_dev,
    load_proposals,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def temporal_iou(gt_boxes, pred_boxes):
    shared = [f for f in pred_boxes if f in gt_boxes]
    if not shared:
        return 0.0
    return float(np.mean([box_iou(gt_boxes[f], pred_boxes[f]) for f in shared]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Q1_DEV))
    ap.add_argument("--out", default="outputs/iclr27_phase5b/audit/geometry")
    args = ap.parse_args()

    rows = load_proposals(Path(args.csv))
    tracks = group_tracks(rows)
    stream, labels_all = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels_all[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)
    known_set = known_ids()

    # GT boxes indexed by image_id -> list of (sid, box)
    gt_by_image = defaultdict(list)
    sid_to_vid = {}
    for r in stream:
        sid = r["sample_id"]
        sid_to_vid[sid] = int(r["video_id"])
        for path, b in zip(r["image_paths"], r["boxes_xyxy"]):
            # image_id lookup happens below via TAO annotation map; boxes are
            # keyed by image_id in gt_track_boxes.
            pass
    # gt_track_boxes already keyed by image_id; rebuild image_id -> list
    for vid, gt_by_track in gb.items():
        for gtid, boxes in gt_by_track.items():
            sid = f"{vid}_{gtid}"
            for img_id, b in boxes.items():
                gt_by_image[img_id].append((sid, b))

    # predicted category distribution
    pred_cats = Counter(int(r["category_id"]) for r in rows)
    known_pred_rows = sum(v for k, v in pred_cats.items() if k in known_set)

    table = []
    gt_overlap = defaultdict(list)   # sid -> pred keys
    gt_same_frame = defaultdict(lambda: defaultdict(int))  # sid -> frame -> n preds
    for key, tr in tracks.items():
        tr_sorted = sorted(tr, key=lambda r: r["frame_id"])
        boxes = {int(r["image_id"]): json.loads(r["bbox_xyxy"])
                 for r in tr_sorted}
        cat = Counter(int(r["category_id"]) for r in tr_sorted)
        pred_cat = cat.most_common(1)[0][0]
        best_sid, best_iou = None, 0.0
        best_shared = 0
        matched_th = {0.3: 0, 0.5: 0, 0.7: 0}
        multi_gt = set()
        per_frame_gt = {}
        any_iou_pos = 0
        for r in tr_sorted:
            img = int(r["image_id"])
            best_frame_sid, best_frame_iou = None, 0.0
            for sid, b in gt_by_image.get(img, []):
                v = box_iou(json.loads(r["bbox_xyxy"]), b)
                if v > 0:
                    any_iou_pos += 1
                if v >= 0.5:
                    multi_gt.add(sid)
                if v > best_frame_iou:
                    best_frame_iou, best_frame_sid = v, sid
                for th in matched_th:
                    if v >= th:
                        matched_th[th] += 1
            per_frame_gt[img] = best_frame_sid
        # best GT track by mean IoU over shared frames
        cands = []
        for sid, boxes_gt in gt_by_image.items():
            pass
        # compute temporal iou for candidate GT tracks
        all_sids = set(s for imgs in gt_by_image.values() for s, _ in imgs)
        sid_boxes = defaultdict(dict)
        for img, lst in gt_by_image.items():
            for sid, b in lst:
                sid_boxes[sid][img] = b
        for sid in all_sids:
            v = temporal_iou(sid_boxes[sid], boxes)
            shared = len(set(boxes) & set(sid_boxes[sid]))
            if v > 0 and shared > 0:
                cands.append((v, shared, sid))
        cands.sort(reverse=True)
        if cands:
            best_iou, best_shared, best_sid = cands[0]
        aligned = key in mapping
        gt_role = (labels[mapping[key]]["protocol_role"] if aligned else None)
        gt_cat = (int(labels[mapping[key]]["ground_truth_category_id"])
                  if aligned else None)
        # category match vs GT for overlapped known GT tracks
        class_match = None
        if aligned and gt_role in ("supported_known", "zero_shot_known"):
            class_match = int(pred_cat) == gt_cat
        n_frames = len(tr_sorted)
        n_frames_gt_box = sum(1 for r in tr_sorted
                              if int(r["image_id"]) in gt_by_image)
        table.append({
            "video_id": int(key[0]), "track_id": int(key[1]),
            "length": n_frames,
            "first_frame": int(tr_sorted[0]["frame_id"]),
            "last_frame": int(tr_sorted[-1]["frame_id"]),
            "mean_score": float(np.mean([r["score"] for r in tr_sorted])),
            "max_score": float(np.max([r["score"] for r in tr_sorted])),
            "first_score": float(tr_sorted[0]["score"]),
            "last_score": float(tr_sorted[-1]["score"]),
            "pred_category": int(pred_cat),
            "pred_category_known": int(pred_cat) in known_set,
            "aligned": int(aligned),
            "gt_sid": best_sid if aligned else None,
            "gt_role": gt_role,
            "gt_category": gt_cat,
            "best_gt_sid_geom": best_sid,
            "best_gt_temporal_iou": round(best_iou, 4),
            "best_gt_shared_frames": best_shared,
            "matched_frames_0.3": matched_th[0.3],
            "matched_frames_0.5": matched_th[0.5],
            "matched_frames_0.7": matched_th[0.7],
            "matched_frame_ratio_0.5": matched_th[0.5] / max(n_frames, 1),
            "n_frames_with_gt_box": n_frames_gt_box,
            "any_iou_pos_frames": any_iou_pos,
            "overlaps_multi_gt_0.5": len(multi_gt),
            "class_match": class_match,
        })
        if best_sid is not None:
            gt_overlap[best_sid].append(key)
        for r in tr_sorted:
            img = int(r["image_id"])
            for sid, b in gt_by_image.get(img, []):
                if box_iou(json.loads(r["bbox_xyxy"]), b) >= 0.5:
                    gt_same_frame[sid][img] += 1

    # fragmentation / duplicate per GT track
    frag = {}
    dup_active = {}
    for sid in sorted(sid_to_vid):
        preds = gt_overlap.get(sid, [])
        frag[sid] = len(preds)
        dup_active[sid] = int(sum(1 for v in gt_same_frame[sid].values()
                                  if v > 1))
    # GT coverage by geometry at 0.5 (track has >=1 matched frame)
    geom_covered = set()
    for row in table:
        if row["matched_frames_0.5"] > 0:
            geom_covered.add(row["best_gt_sid_geom"])
    n_gt_geom_covered = len(geom_covered & set(labels))

    # classification of unaligned tracks
    def classify(row):
        if row["aligned"]:
            return "aligned"
        if row["matched_frames_0.5"] > 0:
            return "fragment_or_duplicate_of_gt"
        if row["any_iou_pos_frames"] > 0:
            return "partial_overlap_annotation_uncertain"
        return "geometry_unmatched"

    for row in table:
        row["forensic_cause"] = classify(row)
    causes = Counter(r["forensic_cause"] for r in table)
    unaligned_causes = Counter(r["forensic_cause"] for r in table
                               if not r["aligned"])
    unaligned_rows = sum(len(tracks[tuple((r["video_id"], r["track_id"]))])
                         for r in table if not r["aligned"])

    summary = {
        "n_gt_tracks": len(labels),
        "n_gt_known": sum(1 for x in labels.values()
                          if x["protocol_role"] in ("supported_known",
                                                    "zero_shot_known")),
        "n_gt_novel": sum(1 for x in labels.values()
                          if x["protocol_role"] == "novel"),
        "n_gt_geom_covered_0.5": n_gt_geom_covered,
        "n_gt_not_geom_covered_0.5": len(labels) - n_gt_geom_covered,
        "aligned_greedy": sum(1 for r in table if r["aligned"]),
        "unaligned": sum(1 for r in table if not r["aligned"]),
        "frame_matched_0.3": sum(r["matched_frames_0.3"] for r in table),
        "frame_matched_0.5": sum(r["matched_frames_0.5"] for r in table),
        "frame_matched_0.7": sum(r["matched_frames_0.7"] for r in table),
        "rows": len(rows),
        "pred_category": {
            "known_rows": known_pred_rows,
            "non_known_rows": len(rows) - known_pred_rows,
            "top_non_known": pred_cats.most_common(10),
        },
        "track_causes_all": dict(causes),
        "track_causes_unaligned": dict(unaligned_causes),
        "unaligned_rows_by_cause": {
            c: sum(len(tracks[(r["video_id"], r["track_id"])])
                   for r in table if not r["aligned"]
                   and r["forensic_cause"] == c)
            for c in causes
        },
        "fragmentation": {
            "gt_tracks_with_preds": len(frag),
            "median_preds_per_gt": float(np.median(list(frag.values())))
            if frag else 0.0,
            "p90_preds_per_gt": float(np.percentile(list(frag.values()), 90))
            if frag else 0.0,
            "max_preds_per_gt": int(max(frag.values())) if frag else 0,
            "gt_with_duplicate_active_frames": sum(1 for v in
                                                   dup_active.values() if v),
            "worst_gt": sorted(frag.items(), key=lambda x: -x[1])[:10],
        },
        "one_pred_to_many_gt": int(sum(1 for r in table
                                       if r["overlaps_multi_gt_0.5"] > 1)),
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "track_forensic_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    (out / "geometry_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
