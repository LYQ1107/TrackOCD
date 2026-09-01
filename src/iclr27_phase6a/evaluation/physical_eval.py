"""Phase 6A physical + coarse semantic evaluation on a proposals CSV.

Input columns (from src/iclr27_phase4p/ovtr_main_eval.py):
  video_id, frame_id, image_id, proposal_local_id, track_id, score,
  bbox_xyxy, category_id, gt_role, gt_iou, gt_category_id, prior_hits
  [+ sem_action, sem_sid when the OVTR eval used score_mode=joint]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.protocol import (
    group_tracks,
    load_gt_tracks_dev,
    load_proposals,
)


def parse_bbox(s: str):
    return np.asarray(json.loads(s), dtype=np.float64)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = ((ax2 - ax1) * (ay2 - ay1)
             + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_proposals(Path(args.csv))
    n_rows = len(rows)
    tracks = defaultdict(list)
    for i, r in enumerate(rows):
        tracks[(int(r["video_id"]), int(r["track_id"]))].append(i)

    lens = [len(v) for v in tracks.values()]
    n_tracks = len(tracks)
    len1 = sum(1 for x in lens if x == 1)
    len2 = sum(1 for x in lens if x <= 2)

    # Frozen Phase5B alignment: greedy one-to-one temporal mean-IoU > 0
    # against the 98 GT dev tracks.
    stream, labels = load_gt_tracks_dev()
    gt_boxes = gt_track_boxes(stream)
    mapping = align_pred_to_gt(group_tracks(rows), gt_boxes)
    sid2pk = {sid: pk for pk, sid in mapping.items()}
    n_aligned_tracks = len(mapping)
    known_tracks = {sid2pk[s] for s in sid2pk
                    if labels[s]["protocol_role"] in
                    ("supported_known", "zero_shot_known")}
    novel_tracks = {sid2pk[s] for s in sid2pk
                    if labels[s]["protocol_role"] == "novel"}
    aligned_tracks = set(mapping.values())

    # Fragmentation: predicted tracks mapping to the same GT track is zero
    # by construction (one-to-one); report the conservative
    # per-(video, GT category) fragmentation as the Phase5B-style proxy.
    by_gt_cat = defaultdict(set)
    for pk, sid in mapping.items():
        vid = int(pk[0])
        cat = int(labels[sid]["ground_truth_category_id"])
        by_gt_cat[(vid, cat)].add(int(pk[1]))
    frag_cats = {k: len(v) for k, v in by_gt_cat.items() if len(v) > 1}
    n_fragmented_cats = len(frag_cats)

    # Duplicate active predictions (Phase5B definition): for each GT track,
    # count frames where >=2 predicted tracks (any track ids) have a box with
    # IoU >= 0.5 against the GT box in that frame.
    dup_frames = 0
    dup_gt_tracks = set()
    per_frame = defaultdict(list)
    for r in rows:
        per_frame[(int(r["video_id"]), int(r["image_id"]))].append(r)
    for vid, gtid in [(int(s.split("_")[0]), int(s.split("_")[1]))
                      for s in labels]:
        gt_track = gt_boxes.get(vid, {}).get(gtid, {})
        for img_id, gt_box in gt_track.items():
            fr = per_frame.get((vid, img_id), [])
            if len(fr) < 2:
                continue
            hit_tracks = set()
            for r in fr:
                b = parse_bbox(r["bbox_xyxy"])
                if iou(b, gt_box) >= 0.5:
                    hit_tracks.add(int(r["track_id"]))
            if len(hit_tracks) >= 2:
                dup_frames += 1
                dup_gt_tracks.add(f"{vid}_{gtid}")
    # semantic fields (present for score_mode=joint)
    has_sem = any("sem_action" in r and r.get("sem_action") for r in rows)
    sem = {}
    if has_sem:
        actions = defaultdict(int)
        n_new = 0
        novel_slots = set()
        slot_tracks = defaultdict(set)
        first_slot_track = {}
        for r in rows:
            a = r.get("sem_action") or ""
            actions[a] += 1
            if a == "new":
                n_new += 1
            sid = int(r["sem_sid"]) if r.get("sem_sid") not in ("", None) else None
            if a in ("new", "existing") and sid is not None:
                novel_slots.add(sid)
                slot_tracks[sid].add((int(r["video_id"]), int(r["track_id"])))
                first_slot_track.setdefault(sid, (int(r["video_id"]), int(r["track_id"])))
        cross_reuse = 0
        for sid, trs in slot_tracks.items():
            if len(trs) > 1:
                cross_reuse += 1
        sem = {
            "has_semantic_fields": True,
            "actions": dict(actions),
            "n_new_actions": n_new,
            "n_novel_slots": len(novel_slots),
            "n_cross_physical_slots": cross_reuse,
            "max_novel_sid": max(novel_slots) if novel_slots else 0,
        }
    else:
        sem = {"has_semantic_fields": False}

    first_score = {}
    for r in rows:
        key = (int(r["video_id"]), int(r["track_id"]))
        if key not in first_score:
            first_score[key] = float(r["score"])
    first_scores = np.asarray(list(first_score.values()), dtype=np.float64)
    known_first = np.asarray(
        [first_score[k] for k in known_tracks if k in first_score],
        dtype=np.float64)
    novel_first = np.asarray(
        [first_score[k] for k in novel_tracks if k in first_score],
        dtype=np.float64)

    result = {
        "n_rows": n_rows,
        "n_tracks": n_tracks,
        "track_len1": len1,
        "track_len1_frac": round(len1 / n_tracks, 4) if n_tracks else 0.0,
        "track_len2_frac": round(len2 / n_tracks, 4) if n_tracks else 0.0,
        "median_track_len": float(np.median(lens)) if lens else 0.0,
        "n_aligned_tracks": n_aligned_tracks,
        "n_known_tracks": len(known_tracks),
        "n_novel_tracks": len(novel_tracks),
        "n_fragmented_gt_categories": n_fragmented_cats,
        "fragmented_gt_categories": {str(k): v for k, v in sorted(
            frag_cats.items())},
        "n_duplicate_active_frames": dup_frames,
        "n_gt_tracks_with_duplicate_active_frames": len(dup_gt_tracks),
        "first_score_mean_all": float(first_scores.mean()) if len(first_scores) else 0.0,
        "first_score_mean_known": float(known_first.mean()) if len(known_first) else 0.0,
        "first_score_mean_novel": float(novel_first.mean()) if len(novel_first) else 0.0,
        "semantic": sem,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
