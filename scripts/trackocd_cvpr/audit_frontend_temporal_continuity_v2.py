#!/usr/bin/env python3
"""Audit anonymous Train continuity on adjacent annotated observations.

The first continuity audit used ``frame_index + 1`` as the definition of a
consecutive pair.  TAO's sealed geometry index is temporally sampled, so that
definition is not valid: adjacent observations of a GT track commonly have a
gap of 15--30 source frames.  This revision compares adjacent observations in
each GT track's annotation order and reports the source-frame gap separately.

Only the sealed category-free physical GT index and anonymous
``train_predicted`` streams are read.  Matching remains per-image,
class-agnostic Hungarian IoU matching.  The prediction-to-GT mapping is held
in memory for aggregate counting and is never written to the receipt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter, defaultdict
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


def _adjacent_continuity_summary(
    gt_frames_by_track: dict[tuple[int, str], list[int]],
    matched_by_gt: dict[tuple[int, str], dict[int, tuple[int, str]]],
) -> dict[str, Any]:
    """Summarize adjacent GT annotation pairs without exposing the mapping."""

    total_pairs = 0
    pairs_with_both_matches = 0
    same_id_pairs = 0
    switch_pairs = 0
    tracks_with_any_adjacent_match = 0
    tracks_with_same_id_pair = 0
    tracks_with_switch_pair = 0
    frame_gap_histogram: Counter[int] = Counter()
    gap_outcomes: dict[int, Counter[str]] = defaultdict(Counter)

    for gt_key, raw_frames in gt_frames_by_track.items():
        frames = sorted(raw_frames)
        matched = matched_by_gt.get(gt_key, {})
        track_total = 0
        track_same = 0
        track_switch = 0
        for left, right in zip(frames, frames[1:]):
            total_pairs += 1
            gap = right - left
            frame_gap_histogram[gap] += 1
            gap_outcomes[gap]["total"] += 1
            if left not in matched or right not in matched:
                gap_outcomes[gap]["missing_match"] += 1
                continue
            pairs_with_both_matches += 1
            gap_outcomes[gap]["both_matched"] += 1
            track_total += 1
            if matched[left] == matched[right]:
                same_id_pairs += 1
                gap_outcomes[gap]["same_pred_id"] += 1
                track_same += 1
            else:
                switch_pairs += 1
                gap_outcomes[gap]["pred_id_switch"] += 1
                track_switch += 1
        if track_total:
            tracks_with_any_adjacent_match += 1
        if track_same:
            tracks_with_same_id_pair += 1
        if track_switch:
            tracks_with_switch_pair += 1

    return {
        "gt_adjacent_annotation_pairs": total_pairs,
        "gt_adjacent_pairs_with_both_stream_matches": pairs_with_both_matches,
        "adjacent_same_pred_id_pairs": same_id_pairs,
        "adjacent_pred_id_switch_pairs": switch_pairs,
        "gt_tracks_with_adjacent_match": tracks_with_any_adjacent_match,
        "gt_tracks_with_adjacent_same_pred_id": tracks_with_same_id_pair,
        "gt_tracks_with_adjacent_pred_id_switch": tracks_with_switch_pair,
        "gt_adjacent_pairs_with_source_gap_one": frame_gap_histogram.get(1, 0),
        "gt_adjacent_pairs_with_source_gap_gt_one": sum(
            count for gap, count in frame_gap_histogram.items() if gap > 1
        ),
        "gt_annotation_source_frame_gap_histogram_top20": [
            {"gap": int(gap), "count": int(count)}
            for gap, count in frame_gap_histogram.most_common(20)
        ],
        "adjacent_pair_outcomes_by_source_gap_top20": [
            {
                "gap": int(gap),
                **{name: int(count) for name, count in gap_outcomes[gap].items()},
            }
            for gap, _count in frame_gap_histogram.most_common(20)
        ],
    }


def audit(gt_path: Path, stream_root: Path, output_path: Path) -> dict[str, Any]:
    gt_by_frame, gt_track_counts = _load_geometry_gt(gt_path)
    stream_paths = _stream_paths(stream_root)
    gt_frames_by_track: dict[tuple[int, str], list[int]] = defaultdict(list)
    for video_id, frames in gt_by_frame.items():
        for frame_index, rows in frames.items():
            for track_id, _box in rows:
                gt_frames_by_track[(video_id, track_id)].append(frame_index)

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

    continuity = _adjacent_continuity_summary(gt_frames_by_track, matched_by_gt)
    total_gt_tracks = sum(len(track_map) for track_map in gt_track_counts.values())
    result = {
        "schema_version": "trackocd.cvpr.frontend-temporal-continuity-audit.v2",
        "status": "COMPLETE_CATEGORY_FREE_FRONTEND_TEMPORAL_CONTINUITY_DIAGNOSTIC_V2",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "videos": len(gt_by_frame),
            "stream_substreams": len(stream_paths),
            "iou_threshold": IOU_THRESHOLD,
            "pair_definition": "adjacent observations in each sealed GT track annotation order",
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
            **continuity,
        },
        "aggregate_geometry": {
            "observation_precision_on_annotated_images": matched_observations / max(predicted_on_gt_images, 1),
            "observation_recall_against_annotated_gt": matched_observations
            / max(sum(len(rows) for frames in gt_by_frame.values() for rows in frames.values()), 1),
            "adjacent_pair_match_rate": continuity["gt_adjacent_pairs_with_both_stream_matches"]
            / max(continuity["gt_adjacent_annotation_pairs"], 1),
            "same_pred_id_rate_given_adjacent_pair_matches": continuity["adjacent_same_pred_id_pairs"]
            / max(continuity["gt_adjacent_pairs_with_both_stream_matches"], 1),
            "pred_id_switch_rate_given_adjacent_pair_matches": continuity["adjacent_pred_id_switch_pairs"]
            / max(continuity["gt_adjacent_pairs_with_both_stream_matches"], 1),
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
            "This revision uses adjacent sampled GT observations rather than a "
            "source-frame gap of exactly one. The prior v1 zero-pair result is "
            "therefore not evidence about fragmentation. The v2 counts are still "
            "category-free frontend diagnostics, not frontend qualification, "
            "OCD/MOT, model selection, or Goal A/B/C acceptance."
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
