#!/usr/bin/env python3
"""Diagnose category-free frontend coverage versus track fragmentation.

The audit uses only the sealed physical GT index and anonymous Train physical
streams.  It performs per-image class-free Hungarian matching, then counts
whether matched observations of one physical GT track occur on consecutive
frames and whether the anonymous predicted track ID changes between those
consecutive matches.  No prediction-to-GT mapping is written to the receipt.
This is a frontend/root-cause diagnostic, not an OCD/MOT metric.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment

from scripts.trackocd_cvpr.audit_predicted_train_geometry import (
    DEFAULT_GT,
    DEFAULT_STREAM_ROOT,
    FORBIDDEN_FIELDS,
    IOU_THRESHOLD,
    PHYSICAL_FIELDS,
    _box_iou_matrix,
    _finite_box,
    _load_geometry_gt,
    _sha256,
    _stream_frame_bases,
    _stream_paths,
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def audit(gt_path: Path, stream_root: Path, output_path: Path) -> dict[str, Any]:
    gt_by_frame, gt_track_counts = _load_geometry_gt(gt_path)
    stream_paths = _stream_paths(stream_root)
    by_frame: dict[tuple[int, int], list[tuple[tuple[int, str], list[float]]]] = defaultdict(list)
    stream_rows = 0
    stream_observations = 0
    frame_bases: dict[Path, dict[int, int]] = {}

    for stream_path in stream_paths:
        bases = _stream_frame_bases(stream_path)
        frame_bases[stream_path] = bases
        with stream_path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict) or set(row) != PHYSICAL_FIELDS:
                    raise ValueError(f"physical stream schema mismatch at {stream_path}:{line_number}")
                if FORBIDDEN_FIELDS & set(row) or row.get("source_split") != "train_predicted":
                    raise ValueError(f"forbidden/non-Train field at {stream_path}:{line_number}")
                video_id = int(row["video_id"])
                if video_id not in bases or video_id not in gt_by_frame:
                    raise ValueError(f"stream video {video_id} has no registered geometry base")
                frame_ids = row["frame_ids"]
                boxes = row["boxes_xyxy"]
                if (
                    not isinstance(frame_ids, list)
                    or not isinstance(boxes, list)
                    or len(frame_ids) != len(boxes)
                    or not frame_ids
                ):
                    raise ValueError(f"invalid physical stream lineage at {stream_path}:{line_number}")
                track_key = (video_id, str(row["physical_track_id"]))
                base = bases[video_id]
                for frame_id, box in zip(frame_ids, boxes):
                    if not _finite_box(box):
                        raise ValueError(f"invalid box at {stream_path}:{line_number}")
                    frame_index = int(frame_id) - base
                    if frame_index in gt_by_frame[video_id]:
                        by_frame[(video_id, frame_index)].append(
                            (track_key, [float(value) for value in box])
                        )
                stream_rows += 1
                stream_observations += len(frame_ids)

    matched_by_gt: dict[tuple[int, str], dict[int, tuple[int, str]]] = defaultdict(dict)
    predicted_on_gt_images = sum(len(rows) for rows in by_frame.values())
    matched_observations = 0

    for (video_id, frame_index), predictions in by_frame.items():
        gt_rows = gt_by_frame.get(video_id, {}).get(frame_index, [])
        if not gt_rows or not predictions:
            continue
        gt_keys = [(video_id, key) for key, _box in gt_rows]
        gt_boxes = [box for _key, box in gt_rows]
        pred_keys = [key for key, _box in predictions]
        pred_boxes = [box for _key, box in predictions]
        matrix = _box_iou_matrix(gt_boxes, pred_boxes)
        weights = matrix.copy()
        weights[weights < IOU_THRESHOLD] = 0.0
        gt_indices, pred_indices = linear_sum_assignment(weights, maximize=True)
        for gt_index, pred_index in zip(gt_indices, pred_indices):
            if weights[gt_index, pred_index] < IOU_THRESHOLD:
                continue
            gt_key = gt_keys[int(gt_index)]
            matched_by_gt[gt_key][frame_index] = pred_keys[int(pred_index)]
            matched_observations += 1

    gt_tracks_with_any_match = 0
    gt_tracks_with_consecutive_match = 0
    gt_tracks_with_consecutive_same_id = 0
    gt_tracks_with_consecutive_id_switch = 0
    consecutive_match_pairs = 0
    consecutive_same_id_pairs = 0
    consecutive_id_switch_pairs = 0
    for frames in matched_by_gt.values():
        if frames:
            gt_tracks_with_any_match += 1
        consecutive_for_track = 0
        same_for_track = 0
        switch_for_track = 0
        for frame_index in sorted(frames):
            if frame_index + 1 not in frames:
                continue
            consecutive_for_track += 1
            consecutive_match_pairs += 1
            if frames[frame_index] == frames[frame_index + 1]:
                same_for_track += 1
                consecutive_same_id_pairs += 1
            else:
                switch_for_track += 1
                consecutive_id_switch_pairs += 1
        if consecutive_for_track:
            gt_tracks_with_consecutive_match += 1
        if same_for_track:
            gt_tracks_with_consecutive_same_id += 1
        if switch_for_track:
            gt_tracks_with_consecutive_id_switch += 1

    total_gt_tracks = sum(len(track_map) for track_map in gt_track_counts.values())
    result = {
        "schema_version": "trackocd.cvpr.frontend-temporal-continuity-audit.v1",
        "status": "COMPLETE_CATEGORY_FREE_FRONTEND_TEMPORAL_CONTINUITY_DIAGNOSTIC",
        "generated_utc": _now(),
        "scope": {
            "videos": len(gt_by_frame),
            "stream_substreams": len(stream_paths),
            "iou_threshold": IOU_THRESHOLD,
            "official_trackeval_or_ocd_metrics": False,
            "qualification_or_model_selection": False,
            "mapping_persisted": False,
        },
        "inputs": {
            "geometry_only_gt_index": str(gt_path.resolve()),
            "geometry_only_gt_index_sha256": _sha256(gt_path),
            "stream_root": str(stream_root.resolve()),
            "stream_sha256_by_path": {
                str(path.relative_to(stream_root)): _sha256(path) for path in stream_paths
            },
        },
        "counts": {
            "predicted_stream_rows": stream_rows,
            "predicted_stream_observations": stream_observations,
            "predicted_observations_on_gt_images": predicted_on_gt_images,
            "gt_tracks": total_gt_tracks,
            "gt_observations": sum(
                len(rows) for frames in gt_by_frame.values() for rows in frames.values()
            ),
            "matched_observations": matched_observations,
            "gt_tracks_with_any_match": gt_tracks_with_any_match,
            "gt_tracks_with_consecutive_match": gt_tracks_with_consecutive_match,
            "gt_tracks_with_consecutive_same_pred_id": gt_tracks_with_consecutive_same_id,
            "gt_tracks_with_consecutive_pred_id_switch": gt_tracks_with_consecutive_id_switch,
            "consecutive_gt_match_pairs": consecutive_match_pairs,
            "consecutive_same_pred_id_pairs": consecutive_same_id_pairs,
            "consecutive_pred_id_switch_pairs": consecutive_id_switch_pairs,
        },
        "aggregate_geometry": {
            "observation_precision_on_annotated_images": matched_observations / max(predicted_on_gt_images, 1),
            "observation_recall_against_annotated_gt": matched_observations
            / max(sum(len(rows) for frames in gt_by_frame.values() for rows in frames.values()), 1),
            "consecutive_match_pair_rate_given_matched_gt": consecutive_match_pairs
            / max(matched_observations, 1),
            "pred_id_switch_rate_given_consecutive_gt_matches": consecutive_id_switch_pairs
            / max(consecutive_match_pairs, 1),
        },
        "boundary": {
            "train_annotation_opened": False,
            "gt_category_fields_read": False,
            "gt_role_or_vocabulary_read": False,
            "val_labels_read": False,
            "test_accessed": False,
            "feature_arrays_read": False,
            "training_started": False,
            "prediction_to_gt_mapping_persisted": False,
        },
        "interpretation": (
            "If consecutive_gt_match_pairs is zero, this audit provides no evidence "
            "that prediction-ID fragmentation is the primary cause of the temporal "
            "gate failure; localization/coverage must be repaired before track "
            "stitching can be scientifically assessed. This is a category-free "
            "diagnostic, not frontend qualification, OCD/MOT, model selection, or "
            "Goal A/B/C acceptance."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--stream-root", type=Path, default=DEFAULT_STREAM_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.geometry_gt.resolve(), args.stream_root.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "scope": result["scope"], "counts": result["counts"], "aggregate_geometry": result["aggregate_geometry"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
