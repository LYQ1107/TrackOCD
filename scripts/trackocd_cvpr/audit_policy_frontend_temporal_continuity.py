#!/usr/bin/env python3
"""Compare category-free temporal continuity on a fixed Train video allowlist.

This is a bounded diagnostic for paired frontend streams.  It uses only the
sealed physical GT geometry index and ``train_predicted`` physical streams,
keeps the per-image matching map in memory, and never writes that map or any
semantic field.  It is deliberately not an official MOT/OCD evaluator.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment

from scripts.trackocd_cvpr.audit_frontend_temporal_continuity_v2 import (
    _adjacent_continuity_summary,
)
from scripts.trackocd_cvpr.audit_predicted_train_geometry import (
    DEFAULT_GT,
    FORBIDDEN_FIELDS,
    IOU_THRESHOLD,
    PHYSICAL_FIELDS,
    _box_iou_matrix,
    _finite_box,
    _load_geometry_gt,
    _sha256,
    _stream_frame_bases,
)


def _read_streams(
    paths: list[Path],
    video_ids: set[int],
    gt_by_frame: dict[int, dict[int, list[tuple[str, list[float]]]]],
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one stream is required")
    by_frame: dict[tuple[int, int], list[tuple[tuple[int, str], list[float]]]] = defaultdict(list)
    seen_tracks: set[tuple[int, str]] = set()
    stream_rows = 0
    stream_observations = 0
    stream_records: list[dict[str, Any]] = []
    observed_videos: set[int] = set()

    for path in paths:
        bases = _stream_frame_bases(path)
        selected_in_path: set[int] = set()
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict) or set(row) != PHYSICAL_FIELDS:
                    raise ValueError(f"physical stream schema mismatch at {path}:{line_number}")
                if FORBIDDEN_FIELDS & set(row) or row.get("source_split") != "train_predicted":
                    raise ValueError(f"forbidden/non-Train row at {path}:{line_number}")
                video_id = int(row["video_id"])
                if video_id not in video_ids:
                    continue
                if video_id not in bases or video_id not in gt_by_frame:
                    raise ValueError(f"stream video {video_id} has no geometry base")
                frame_ids = row["frame_ids"]
                boxes = row["boxes_xyxy"]
                image_paths = row["image_paths"]
                quality = row["quality"]
                if (
                    not isinstance(frame_ids, list)
                    or not isinstance(boxes, list)
                    or not isinstance(image_paths, list)
                    or not isinstance(quality, list)
                    or len(frame_ids) != len(boxes)
                    or len(frame_ids) != len(image_paths)
                    or len(frame_ids) != len(quality)
                    or not frame_ids
                    or len(set(frame_ids)) != len(frame_ids)
                ):
                    raise ValueError(f"invalid physical lineage at {path}:{line_number}")
                track_key = (video_id, str(row["physical_track_id"]))
                if track_key in seen_tracks:
                    raise ValueError(f"duplicate physical track {track_key}")
                seen_tracks.add(track_key)
                selected_in_path.add(video_id)
                base = bases[video_id]
                for frame_id, box in zip(frame_ids, boxes):
                    if not _finite_box(box):
                        raise ValueError(f"invalid box at {path}:{line_number}")
                    frame_index = int(frame_id) - base
                    if frame_index in gt_by_frame[video_id]:
                        by_frame[(video_id, frame_index)].append(
                            (track_key, [float(value) for value in box])
                        )
                stream_rows += 1
                stream_observations += len(frame_ids)
        overlap = observed_videos & selected_in_path
        if overlap:
            raise ValueError(f"video IDs appear in multiple streams: {sorted(overlap)}")
        observed_videos.update(selected_in_path)
        stream_records.append({
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "selected_videos": sorted(selected_in_path),
        })

    if observed_videos != video_ids:
        raise ValueError(f"stream videos {sorted(observed_videos)} != expected {sorted(video_ids)}")

    matched_by_gt: dict[tuple[int, str], dict[int, tuple[int, str]]] = defaultdict(dict)
    matched_observations = 0
    predicted_on_gt_images = sum(len(rows) for rows in by_frame.values())
    for (video_id, frame_index), predictions in by_frame.items():
        gt_rows = gt_by_frame[video_id].get(frame_index, [])
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

    gt_frames_by_track: dict[tuple[int, str], list[int]] = defaultdict(list)
    for video_id, frames in gt_by_frame.items():
        for frame_index, rows in frames.items():
            for track_id, _box in rows:
                gt_frames_by_track[(video_id, track_id)].append(frame_index)
    continuity = _adjacent_continuity_summary(gt_frames_by_track, matched_by_gt)
    gt_observations = sum(
        len(rows) for frames in gt_by_frame.values() for rows in frames.values()
    )
    return {
        "streams": stream_records,
        "stream_rows": stream_rows,
        "stream_observations": stream_observations,
        "predicted_observations_on_gt_images": predicted_on_gt_images,
        "matched_observations": matched_observations,
        "gt_tracks": len(gt_frames_by_track),
        "gt_observations": gt_observations,
        "continuity": continuity,
        "aggregate": {
            "observation_precision_on_annotated_images": matched_observations / max(predicted_on_gt_images, 1),
            "observation_recall_against_annotated_gt": matched_observations / max(gt_observations, 1),
            "adjacent_pair_match_rate": continuity["gt_adjacent_pairs_with_both_stream_matches"]
            / max(continuity["gt_adjacent_annotation_pairs"], 1),
            "same_pred_id_rate_given_adjacent_pair_matches": continuity["adjacent_same_pred_id_pairs"]
            / max(continuity["gt_adjacent_pairs_with_both_stream_matches"], 1),
            "pred_id_switch_rate_given_adjacent_pair_matches": continuity["adjacent_pred_id_switch_pairs"]
            / max(continuity["gt_adjacent_pairs_with_both_stream_matches"], 1),
        },
    }


def audit(
    *,
    geometry_gt: Path,
    baseline_streams: list[Path],
    candidate_streams: list[Path],
    video_ids: list[int],
    output: Path,
) -> dict[str, Any]:
    selected = set(video_ids)
    if not selected or len(selected) != len(video_ids):
        raise ValueError("video IDs must be unique and non-empty")
    all_gt_by_frame, _gt_counts = _load_geometry_gt(geometry_gt)
    gt_by_frame = {video_id: all_gt_by_frame[video_id] for video_id in selected}
    baseline = _read_streams(baseline_streams, selected, gt_by_frame)
    candidate = _read_streams(candidate_streams, selected, gt_by_frame)
    result = {
        "schema_version": "trackocd.cvpr.policy-frontend-temporal-continuity.v1",
        "status": "COMPLETE_CATEGORY_FREE_POLICY_FRONTEND_TEMPORAL_CONTINUITY",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "videos": sorted(selected),
            "iou_threshold": IOU_THRESHOLD,
            "pair_definition": "adjacent observations in each sealed GT track annotation order",
            "official_trackeval_or_ocd_metrics": False,
            "frontend_qualification": False,
            "model_selection": False,
            "mapping_persisted": False,
        },
        "inputs": {
            "geometry_only_gt_index": str(geometry_gt.resolve()),
            "geometry_only_gt_index_sha256": _sha256(geometry_gt),
        },
        "baseline": baseline,
        "candidate": candidate,
        "deltas_candidate_minus_baseline": {
            "matched_observations": candidate["matched_observations"] - baseline["matched_observations"],
            "observation_recall": (
                candidate["aggregate"]["observation_recall_against_annotated_gt"]
                - baseline["aggregate"]["observation_recall_against_annotated_gt"]
            ),
            "adjacent_pair_match_rate": (
                candidate["aggregate"]["adjacent_pair_match_rate"]
                - baseline["aggregate"]["adjacent_pair_match_rate"]
            ),
            "same_id_rate_given_matches": (
                candidate["aggregate"]["same_pred_id_rate_given_adjacent_pair_matches"]
                - baseline["aggregate"]["same_pred_id_rate_given_adjacent_pair_matches"]
            ),
            "switch_rate_given_matches": (
                candidate["aggregate"]["pred_id_switch_rate_given_adjacent_pair_matches"]
                - baseline["aggregate"]["pred_id_switch_rate_given_adjacent_pair_matches"]
            ),
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
            "This paired category-free continuity diagnostic tests whether a score-contract "
            "candidate preserves identities across adjacent sealed GT observations. It is "
            "not an official MOT/OCD metric or frontend qualification."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--baseline-stream", type=Path, action="append", required=True)
    parser.add_argument("--candidate-stream", type=Path, action="append", required=True)
    parser.add_argument("--video-id", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        geometry_gt=args.geometry_gt.resolve(),
        baseline_streams=[path.resolve() for path in args.baseline_stream],
        candidate_streams=[path.resolve() for path in args.candidate_stream],
        video_ids=args.video_id,
        output=args.output.resolve(),
    )
    print(json.dumps({
        "status": result["status"],
        "baseline": result["baseline"]["aggregate"],
        "candidate": result["candidate"]["aggregate"],
        "deltas": result["deltas_candidate_minus_baseline"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
