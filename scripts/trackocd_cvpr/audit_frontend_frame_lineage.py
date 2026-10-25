#!/usr/bin/env python3
"""Audit anonymous Train stream frame IDs against image-path lineage.

Dataset-specific image filenames are treated as opaque frame keys: some TAO
videos use ``frameNNNN.jpg`` while others use timestamp-like names.  For each
video this audit checks that the same image path always maps to one local frame
and that one local frame never maps to multiple image paths.  It also checks
strict per-track frame order and the geometry-only video set.  No pixels or
semantic fields are read and only aggregate counts are emitted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.trackocd_cvpr.audit_predicted_train_geometry import (
    DEFAULT_GT,
    DEFAULT_STREAM_ROOT,
    FORBIDDEN_FIELDS,
    PHYSICAL_FIELDS,
    _load_geometry_gt,
    _sha256,
    _stream_frame_bases,
    _stream_paths,
)


def audit(gt_path: Path, stream_root: Path, output_path: Path) -> dict[str, Any]:
    gt_by_frame, _gt_track_counts = _load_geometry_gt(gt_path)
    gt_ranges = {
        video_id: (min(frames), max(frames), len(frames))
        for video_id, frames in gt_by_frame.items()
    }
    stream_paths = _stream_paths(stream_root)

    global_bases: dict[int, int] = {}
    stream_hashes: dict[str, str] = {}
    for stream_path in stream_paths:
        stream_hashes[str(stream_path.relative_to(stream_root))] = _sha256(stream_path)
        for video_id, base in _stream_frame_bases(stream_path).items():
            global_bases[video_id] = min(global_bases.get(video_id, base), base)

    stream_rows = 0
    observations = 0
    path_count = 0
    unique_paths: dict[tuple[int, str], int] = {}
    unique_frames: dict[tuple[int, int], str] = {}
    path_frame_conflicts = 0
    frame_path_conflicts = 0
    strict_order_violations = 0
    missing_gt_video_rows = 0
    local_min: dict[int, int] = {}
    local_max: dict[int, int] = {}
    observed_frames_by_video: defaultdict[int, set[int]] = defaultdict(set)

    for stream_path in stream_paths:
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
                if video_id not in global_bases:
                    raise ValueError(f"no global frame base for video {video_id}")
                if video_id not in gt_ranges:
                    missing_gt_video_rows += 1
                frame_ids = row["frame_ids"]
                image_paths = row["image_paths"]
                if (
                    not isinstance(frame_ids, list)
                    or not isinstance(image_paths, list)
                    or len(frame_ids) != len(image_paths)
                ):
                    raise ValueError(f"frame/path length mismatch at {stream_path}:{line_number}")
                local_ids = [int(value) - global_bases[video_id] for value in frame_ids]
                if any(right <= left for left, right in zip(local_ids, local_ids[1:])):
                    strict_order_violations += 1
                for local_frame, image_path in zip(local_ids, image_paths):
                    path_count += 1
                    path_key = (video_id, str(image_path))
                    if path_key in unique_paths:
                        if unique_paths[path_key] != local_frame:
                            path_frame_conflicts += 1
                    else:
                        unique_paths[path_key] = local_frame
                    frame_key = (video_id, local_frame)
                    if frame_key in unique_frames:
                        if unique_frames[frame_key] != str(image_path):
                            frame_path_conflicts += 1
                    else:
                        unique_frames[frame_key] = str(image_path)
                    observed_frames_by_video[video_id].add(local_frame)
                    local_min[video_id] = min(local_min.get(video_id, local_frame), local_frame)
                    local_max[video_id] = max(local_max.get(video_id, local_frame), local_frame)
                stream_rows += 1
                observations += len(frame_ids)

    video_set_mismatches = len(set(gt_ranges) ^ set(local_min))
    gt_annotated_frames = sum(len(frames) for frames in gt_by_frame.values())
    gt_annotated_frames_with_path = sum(
        len(set(frames) & observed_frames_by_video.get(video_id, set()))
        for video_id, frames in gt_by_frame.items()
    )
    videos_with_any_annotated_frame_path = sum(
        bool(set(frames) & observed_frames_by_video.get(video_id, set()))
        for video_id, frames in gt_by_frame.items()
    )
    videos_with_all_annotated_frame_paths = sum(
        set(frames).issubset(observed_frames_by_video.get(video_id, set()))
        for video_id, frames in gt_by_frame.items()
    )
    result = {
        "schema_version": "trackocd.cvpr.frontend-frame-lineage-audit.v2",
        "status": "COMPLETE_CATEGORY_FREE_FRONTEND_FRAME_LINEAGE_AUDIT",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "videos_in_geometry_gt": len(gt_ranges),
            "stream_substreams": len(stream_paths),
            "pixels_read": False,
            "semantic_fields_read": False,
            "official_trackeval_or_ocd_metrics": False,
        },
        "inputs": {
            "geometry_only_gt_index": str(gt_path.resolve()),
            "geometry_only_gt_index_sha256": _sha256(gt_path),
            "stream_root": str(stream_root.resolve()),
            "stream_sha256_by_path": stream_hashes,
        },
        "counts": {
            "predicted_stream_rows": stream_rows,
            "predicted_stream_observations": observations,
            "image_paths_checked": path_count,
            "unique_video_image_paths": len(unique_paths),
            "image_path_frame_conflicts": path_frame_conflicts,
            "frame_image_path_conflicts": frame_path_conflicts,
            "strict_frame_order_violations": strict_order_violations,
            "missing_geometry_gt_video_rows": missing_gt_video_rows,
            "video_set_mismatches": video_set_mismatches,
            "geometry_gt_annotated_frames": gt_annotated_frames,
            "geometry_gt_annotated_frames_with_stream_path": gt_annotated_frames_with_path,
            "videos_with_any_annotated_frame_path": videos_with_any_annotated_frame_path,
            "videos_with_all_annotated_frame_paths": videos_with_all_annotated_frame_paths,
            "annotated_frame_path_coverage": gt_annotated_frames_with_path / max(gt_annotated_frames, 1),
        },
        "ranges": {
            "geometry_gt_frame_min": min(minimum for minimum, _maximum, _count in gt_ranges.values()),
            "geometry_gt_frame_max": max(maximum for _minimum, maximum, _count in gt_ranges.values()),
            "stream_local_frame_min": min(local_min.values()),
            "stream_local_frame_max": max(local_max.values()),
        },
        "boundary": {
            "train_annotation_opened": False,
            "gt_category_fields_read": False,
            "gt_role_or_vocabulary_read": False,
            "val_labels_read": False,
            "test_accessed": False,
            "feature_arrays_read": False,
            "training_started": False,
        },
        "interpretation": (
            "Zero image_path_frame_conflicts and zero frame_image_path_conflicts "
            "support the anonymous video-local frame lineage used by the geometry "
            "audits; dataset-specific filenames are intentionally opaque. This "
            "does not by itself prove the external GT image-index join beyond the "
            "sealed geometry contract. It is not frontend qualification, OCD/MOT, "
            "model selection, or Goal A/B/C acceptance."
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
    print(json.dumps({"status": result["status"], "scope": result["scope"], "counts": result["counts"], "ranges": result["ranges"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
