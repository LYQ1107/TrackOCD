#!/usr/bin/env python3
"""Audit a predeclared small Train frontend stream against geometry-only GT.

This is a bounded frontend smoke, not a qualification or OCD/MOT metric.  It
reuses the category-free matching primitives from the full Train geometry
audit, reads only the sealed physical GT index, and never writes a
prediction-to-GT mapping or opens semantic fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.trackocd_cvpr.audit_predicted_train_geometry import (
    FORBIDDEN_FIELDS,
    PHYSICAL_FIELDS,
    _finalize_stream,
    _finite_box,
    _load_geometry_gt,
    _new_stream_state,
    _now,
    _sha256,
    _stream_frame_bases,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GT = ROOT / "outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v10_physical_GT/private_exact_joint_proof_or_assignment.json"


def audit(gt_path: Path, stream_path: Path, output_path: Path, video_ids: list[int]) -> dict:
    expected_videos = set(video_ids)
    if not expected_videos:
        raise ValueError("at least one predeclared video ID is required")
    gt_by_frame, gt_track_counts = _load_geometry_gt(gt_path)
    frame_bases = _stream_frame_bases(stream_path)
    state = _new_stream_state()
    digest = _sha256(stream_path)
    seen_tracks: set[tuple[int, str]] = set()

    with stream_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict) or set(row) != PHYSICAL_FIELDS:
                raise ValueError(f"physical stream schema mismatch at line {line_number}")
            if FORBIDDEN_FIELDS & set(row) or row.get("source_split") != "train_predicted":
                raise ValueError(f"forbidden/non-Train field at line {line_number}")
            video_id = int(row["video_id"])
            if video_id not in expected_videos:
                raise ValueError(f"unexpected video {video_id} at line {line_number}")
            track_key = (video_id, str(row["physical_track_id"]))
            if track_key in seen_tracks:
                raise ValueError(f"duplicate physical track {track_key}")
            seen_tracks.add(track_key)
            frame_ids = row["frame_ids"]
            boxes = row["boxes_xyxy"]
            image_paths = row["image_paths"]
            if (
                not isinstance(frame_ids, list)
                or not isinstance(boxes, list)
                or not isinstance(image_paths, list)
                or len(frame_ids) != len(boxes)
                or len(frame_ids) != len(image_paths)
                or not frame_ids
                or len(set(frame_ids)) != len(frame_ids)
            ):
                raise ValueError(f"invalid stream lineage at line {line_number}")
            state["videos"].add(video_id)
            state["tracks"][track_key] = len(frame_ids)
            for frame_id, box in zip(frame_ids, boxes):
                if not _finite_box(box):
                    raise ValueError(f"invalid box at line {line_number}")
                frame_index = int(frame_id) - frame_bases[video_id]
                if frame_index < 0:
                    raise ValueError(f"frame precedes video base at line {line_number}")
                if frame_index in gt_by_frame.get(video_id, {}):
                    state["by_frame"][(video_id, frame_index)].append(
                        (track_key, [float(value) for value in box])
                    )
                    state["annotated_observations"] += 1
            state["rows"] += 1
            state["observations"] += len(frame_ids)

    if state["videos"] != expected_videos:
        raise ValueError(f"stream videos {sorted(state['videos'])} != expected {sorted(expected_videos)}")
    geometry = _finalize_stream(state, gt_by_frame, gt_track_counts)
    predicted_on_gt = int(geometry["predicted_observations_on_gt_images"])
    gt_observations = int(geometry["gt_observations"])
    predicted_tracks = int(geometry["predicted_tracks"])
    gt_tracks = int(geometry["gt_tracks"])
    result = {
        "schema_version": "trackocd.cvpr.frontend-geometry-smoke-audit.v1",
        "status": "COMPLETE_CATEGORY_FREE_FRONTEND_GEOMETRY_SMOKE",
        "generated_utc": _now(),
        "scope": {
            "predeclared_video_ids": sorted(expected_videos),
            "videos_processed": len(state["videos"]),
            "iou_threshold": 0.5,
            "official_trackeval_or_ocd_metrics": False,
            "qualification_or_model_selection": False,
        },
        "inputs": {
            "geometry_only_gt_index": str(gt_path.resolve()),
            "geometry_only_gt_index_sha256": _sha256(gt_path),
            "physical_stream": str(stream_path.resolve()),
            "physical_stream_sha256": digest,
        },
        "counts": geometry,
        "aggregate_geometry": {
            **geometry,
            "observation_precision_on_annotated_images": geometry["per_image_matched_observations"] / max(predicted_on_gt, 1),
            "observation_recall_against_annotated_gt": geometry["per_image_matched_observations"] / max(gt_observations, 1),
            "reliable_predicted_track_fraction_all": geometry["reliable_predicted_tracks"] / max(predicted_tracks, 1),
            "reliable_gt_track_recall": geometry["reliable_gt_tracks"] / max(gt_tracks, 1),
        },
        "boundary": {
            "train_annotation_opened": False,
            "gt_category_fields_read": False,
            "gt_role_or_vocabulary_read": False,
            "native_evaluator_rows_read": False,
            "val_labels_read": False,
            "test_accessed": False,
            "feature_arrays_read": False,
            "training_started": False,
            "prediction_to_gt_track_mapping_persisted": False,
        },
        "interpretation": "Bounded category-free Train frontend smoke only; not frontend qualification, OCD/MOT, model selection, or Goal A/B/C acceptance.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-id", type=int, action="append", required=True)
    args = parser.parse_args()
    result = audit(args.geometry_gt.resolve(), args.stream.resolve(), args.output.resolve(), args.video_id)
    print(json.dumps({"status": result["status"], "scope": result["scope"], "aggregate_geometry": result["aggregate_geometry"], "boundary": result["boundary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
