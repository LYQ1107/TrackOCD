"""Phase 5B stream audit: reproduce Phase 5A counts and basic structure.

Reads outputs/iclr27_phase4q/q1_long/proposals_dev.csv and the frozen dev GT
alignment, and reports: rows, videos, annotated frames, physical tracks,
aligned/unaligned, track lengths, scores, duplicates, track-ID namespace.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Q1_DEV))
    ap.add_argument("--out", default="outputs/iclr27_phase5b/audit/counts")
    args = ap.parse_args()

    rows = load_proposals(Path(args.csv))
    tracks = group_tracks(rows)
    stream, labels_all = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels_all[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)

    lens = np.array([len(tr) for tr in tracks.values()])
    aligned_keys = set(mapping)
    unaligned = [k for k in tracks if k not in aligned_keys]
    aligned_lens = np.array([len(tracks[k]) for k in aligned_keys])
    unaligned_lens = np.array([len(tracks[k]) for k in unaligned])

    # per-track score summaries
    def score_stats(keys):
        first, mean, mx, med, last = [], [], [], [], []
        for k in keys:
            tr = sorted(tracks[k], key=lambda r: r["frame_id"])
            sc = [r["score"] for r in tr]
            first.append(sc[0]); mean.append(float(np.mean(sc)))
            mx.append(max(sc)); med.append(float(np.median(sc)))
            last.append(sc[-1])
        return {
            "first_mean": float(np.mean(first)),
            "mean_mean": float(np.mean(mean)),
            "max_mean": float(np.mean(mx)),
            "median_mean": float(np.mean(med)),
            "last_mean": float(np.mean(last)),
        }

    # duplicate rows
    exact = Counter()
    for r in rows:
        exact[(r["video_id"], r["frame_id"], r["track_id"],
               r["bbox_xyxy"])] += 1
    n_exact_dup_rows = sum(v - 1 for v in exact.values() if v > 1)
    # near duplicates: same video/frame, different track_id, bbox IoU >= 0.5
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[(r["video_id"], r["frame_id"])].append(r)
    near = 0
    near_pairs = 0
    for fr, rs in by_frame.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if rs[i]["track_id"] == rs[j]["track_id"]:
                    continue
                b1 = json.loads(rs[i]["bbox_xyxy"])
                b2 = json.loads(rs[j]["bbox_xyxy"])
                if box_iou(b1, b2) >= 0.5:
                    near_pairs += 1
                    near += 1
    # track id namespace
    vids_by_tid = defaultdict(set)
    for r in rows:
        vids_by_tid[r["track_id"]].add(r["video_id"])
    tid_multi_video = sum(1 for v in vids_by_tid.values() if len(v) > 1)

    summary = {
        "rows": len(rows),
        "videos": len(set(r["video_id"] for r in rows)),
        "annotated_frames": len(set((r["video_id"], r["frame_id"]) for r in rows)),
        "unique_images": len(set(r["image_id"] for r in rows)),
        "tracks": len(tracks),
        "tracks_video_track_unique": len(tracks),
        "track_id_values_multi_video": tid_multi_video,
        "aligned_gt_tracks": len(mapping),
        "aligned_known": sum(1 for sid in mapping.values()
                             if labels[sid]["protocol_role"]
                             in ("supported_known", "zero_shot_known")),
        "aligned_novel": sum(1 for sid in mapping.values()
                             if labels[sid]["protocol_role"] == "novel"),
        "unaligned_tracks": len(unaligned),
        "track_len_mean": float(lens.mean()),
        "track_len_median": float(np.median(lens)),
        "track_len_p90": float(np.percentile(lens, 90)),
        "track_len_p95": float(np.percentile(lens, 95)),
        "track_len_max": int(lens.max()),
        "aligned_len_mean": float(aligned_lens.mean()),
        "aligned_len_median": float(np.median(aligned_lens)),
        "unaligned_len_mean": float(unaligned_lens.mean()),
        "unaligned_len_median": float(np.median(unaligned_lens)),
        "track_len_buckets": {
            "1": int((lens == 1).sum()),
            "2": int((lens == 2).sum()),
            "3": int((lens == 3).sum()),
            "4_5": int(((lens >= 4) & (lens <= 5)).sum()),
            "6_10": int(((lens >= 6) & (lens <= 10)).sum()),
            "11_20": int(((lens >= 11) & (lens <= 20)).sum()),
            "21_50": int(((lens >= 21) & (lens <= 50)).sum()),
            "gt50": int((lens > 50).sum()),
        },
        "score": {
            "aligned": score_stats(aligned_keys),
            "unaligned": score_stats(unaligned),
            "known": score_stats([k for k in aligned_keys
                                  if labels[mapping[k]]["protocol_role"]
                                  in ("supported_known", "zero_shot_known")]),
            "novel": score_stats([k for k in aligned_keys
                                  if labels[mapping[k]]["protocol_role"]
                                  == "novel"]),
        },
        "duplicates": {
            "exact_duplicate_rows": n_exact_dup_rows,
            "near_duplicate_row_hits": near,
            "near_duplicate_pairs": near_pairs,
        },
        "role_counts": dict(Counter(r["gt_role"] for r in rows)),
        "gt_counts": {
            "total": len(labels),
            "known": sum(1 for x in labels.values()
                         if x["protocol_role"] in ("supported_known",
                                                   "zero_shot_known")),
            "novel": sum(1 for x in labels.values()
                         if x["protocol_role"] == "novel"),
            "novel_categories": len({x["ground_truth_category_id"]
                                     for x in labels.values()
                                     if x["protocol_role"] == "novel"}),
        },
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "counts.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("score", "track_len_buckets")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
