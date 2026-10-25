#!/usr/bin/env python3
"""Measure category-free geometry sensitivity to the IoU match threshold.

This is a bounded evaluator-side diagnostic over the sealed physical GT index
and the anonymous Train frontend stream.  It reads no semantic fields, pixels,
features, Val/Test labels, or vocabulary, and persists only aggregate counts.
The per-frame assignment is kept in memory and no prediction-to-GT mapping is
written.
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
    PHYSICAL_FIELDS,
    _box_iou_matrix,
    _finite_box,
    _load_geometry_gt,
    _sha256,
    _stream_frame_bases,
    _stream_paths,
)


THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _threshold_match_counts(
    matrix,
    gt_keys: list[tuple[int, str]],
    pred_keys: list[tuple[int, str]],
    threshold: float,
) -> tuple[int, set[tuple[int, str]], set[tuple[int, str]]]:
    """Return matched count and matched keys for one IoU threshold."""

    weights = matrix.copy()
    weights[weights < threshold] = 0.0
    gt_indices, pred_indices = linear_sum_assignment(weights, maximize=True)
    matched_gt: set[tuple[int, str]] = set()
    matched_pred: set[tuple[int, str]] = set()
    matched = 0
    for gt_index, pred_index in zip(gt_indices, pred_indices):
        if weights[gt_index, pred_index] < threshold:
            continue
        matched += 1
        matched_gt.add(gt_keys[int(gt_index)])
        matched_pred.add(pred_keys[int(pred_index)])
    return matched, matched_gt, matched_pred


def audit(gt_path: Path, stream_root: Path, output_path: Path) -> dict[str, Any]:
    gt_by_frame, gt_track_counts = _load_geometry_gt(gt_path)
    stream_paths = _stream_paths(stream_root)
    by_frame: dict[tuple[int, int], list[tuple[tuple[int, str], list[float]]]] = defaultdict(list)
    stream_rows = 0
    stream_observations = 0
    stream_hashes: dict[str, str] = {}

    for stream_path in stream_paths:
        stream_hashes[str(stream_path.relative_to(stream_root))] = _sha256(stream_path)
        bases = _stream_frame_bases(stream_path)
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
                frame_ids = row["frame_ids"]
                boxes = row["boxes_xyxy"]
                if (
                    video_id not in bases
                    or video_id not in gt_by_frame
                    or not isinstance(frame_ids, list)
                    or not isinstance(boxes, list)
                    or len(frame_ids) != len(boxes)
                    or not frame_ids
                ):
                    raise ValueError(f"invalid physical stream lineage at {stream_path}:{line_number}")
                track_key = (video_id, str(row["physical_track_id"]))
                for frame_id, box in zip(frame_ids, boxes):
                    if not _finite_box(box):
                        raise ValueError(f"invalid box at {stream_path}:{line_number}")
                    frame_index = int(frame_id) - bases[video_id]
                    if frame_index in gt_by_frame[video_id]:
                        by_frame[(video_id, frame_index)].append(
                            (track_key, [float(value) for value in box])
                        )
                stream_rows += 1
                stream_observations += len(frame_ids)

    gt_observations = sum(len(rows) for frames in gt_by_frame.values() for rows in frames.values())
    predicted_on_gt_images = sum(len(rows) for rows in by_frame.values())
    threshold_rows = []
    for threshold in THRESHOLDS:
        matched = 0
        matched_gt: set[tuple[int, str]] = set()
        matched_pred: set[tuple[int, str]] = set()
        for (video_id, frame_index), predictions in by_frame.items():
            gt_rows = gt_by_frame.get(video_id, {}).get(frame_index, [])
            if not gt_rows or not predictions:
                continue
            gt_keys = [(video_id, key) for key, _box in gt_rows]
            gt_boxes = [box for _key, box in gt_rows]
            pred_keys = [key for key, _box in predictions]
            pred_boxes = [box for _key, box in predictions]
            count, frame_gt, frame_pred = _threshold_match_counts(
                _box_iou_matrix(gt_boxes, pred_boxes), gt_keys, pred_keys, threshold
            )
            matched += count
            matched_gt.update(frame_gt)
            matched_pred.update(frame_pred)
        threshold_rows.append(
            {
                "iou_threshold": threshold,
                "matched_observations": matched,
                "observation_precision_on_annotated_images": matched / max(predicted_on_gt_images, 1),
                "observation_recall_against_annotated_gt": matched / max(gt_observations, 1),
                "gt_tracks_with_match": len(matched_gt),
                "gt_track_recall": len(matched_gt) / max(sum(len(v) for v in gt_track_counts.values()), 1),
                "predicted_tracks_with_match": len(matched_pred),
            }
        )

    result = {
        "schema_version": "trackocd.cvpr.frontend-iou-sensitivity-audit.v1",
        "status": "COMPLETE_CATEGORY_FREE_FRONTEND_IOU_SENSITIVITY_DIAGNOSTIC",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "videos": len(gt_by_frame),
            "stream_substreams": len(stream_paths),
            "thresholds": list(THRESHOLDS),
            "official_trackeval_or_ocd_metrics": False,
            "qualification_or_model_selection": False,
            "mapping_persisted": False,
        },
        "inputs": {
            "geometry_only_gt_index": str(gt_path.resolve()),
            "geometry_only_gt_index_sha256": _sha256(gt_path),
            "stream_root": str(stream_root.resolve()),
            "stream_sha256_by_path": stream_hashes,
        },
        "counts": {
            "predicted_stream_rows": stream_rows,
            "predicted_stream_observations": stream_observations,
            "predicted_observations_on_gt_images": predicted_on_gt_images,
            "gt_observations": gt_observations,
            "gt_tracks": sum(len(v) for v in gt_track_counts.values()),
        },
        "threshold_results": threshold_rows,
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
            "Threshold sensitivity is a category-free geometry diagnostic only. "
            "It can separate severe localization from threshold sensitivity, "
            "but it does not establish a legal frontend qualification, an "
            "OCD/MOT metric, model selection, or Goal A/B/C acceptance."
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
    print(json.dumps({"status": result["status"], "counts": result["counts"], "threshold_results": result["threshold_results"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
