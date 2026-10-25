#!/usr/bin/env python3
"""Compare bounded COVTrack score-contract smokes on geometry only.

The compared streams share the same two approved anonymous Train videos and
the same sealed category-free physical GT index.  This audit never opens
semantic fields, native evaluator rows, pixels, or feature arrays, and never
persists a prediction-to-GT mapping.  It is a mechanism diagnostic only; it
does not qualify a frontend or produce an official OCD/MOT metric.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.trackocd_cvpr.audit_predicted_train_geometry import (
    DEFAULT_GT,
    FORBIDDEN_FIELDS,
    PHYSICAL_FIELDS,
    _finalize_stream,
    _finite_box,
    _load_geometry_gt,
    _new_stream_state,
    _sha256,
    _stream_frame_bases,
)


def _read_stream(path: Path, gt_by_frame: dict[int, dict[int, list[Any]]], gt_track_counts: dict[int, dict[str, int]]) -> dict[str, Any]:
    bases = _stream_frame_bases(path)
    state = _new_stream_state()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict) or set(row) != PHYSICAL_FIELDS:
                raise ValueError(f"physical stream schema mismatch at {path}:{line_number}")
            if FORBIDDEN_FIELDS & set(row) or row.get("source_split") != "train_predicted":
                raise ValueError(f"forbidden/non-Train field at {path}:{line_number}")
            video_id = int(row["video_id"])
            if video_id not in gt_track_counts or video_id not in bases:
                raise ValueError(f"stream video {video_id} is outside the approved geometry index")
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
            ):
                raise ValueError(f"invalid physical lineage at {path}:{line_number}")
            track_key = (video_id, str(row["physical_track_id"]))
            if track_key in state["tracks"]:
                raise ValueError(f"duplicate physical track {track_key}")
            state["tracks"][track_key] = len(frame_ids)
            state["videos"].add(video_id)
            for frame_id, box in zip(frame_ids, boxes):
                if not _finite_box(box):
                    raise ValueError(f"invalid box at {path}:{line_number}")
                frame_index = int(frame_id) - bases[video_id]
                if frame_index in gt_by_frame.get(video_id, {}):
                    state["by_frame"][(video_id, frame_index)].append(
                        (track_key, [float(value) for value in box])
                    )
                    state["annotated_observations"] += 1
            state["rows"] += 1
            state["observations"] += len(frame_ids)
    geometry = _finalize_stream(state, gt_by_frame, gt_track_counts)
    geometry["observation_precision_on_annotated_images"] = geometry["per_image_matched_observations"] / max(
        geometry["predicted_observations_on_gt_images"], 1
    )
    geometry["observation_recall_against_gt"] = geometry["per_image_matched_observations"] / max(
        geometry["gt_observations"], 1
    )
    geometry["reliable_gt_track_recall"] = geometry["reliable_gt_tracks"] / max(geometry["gt_tracks"], 1)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "videos": sorted(int(value) for value in state["videos"]),
        "rows": int(state["rows"]),
        "observations": int(state["observations"]),
        "geometry": geometry,
    }


def audit(gt_path: Path, streams: dict[str, Path], output_path: Path) -> dict[str, Any]:
    gt_by_frame, gt_track_counts = _load_geometry_gt(gt_path)
    records = {name: _read_stream(path, gt_by_frame, gt_track_counts) for name, path in streams.items()}
    video_sets = {tuple(record["videos"]) for record in records.values()}
    if len(video_sets) != 1:
        raise ValueError(f"mechanism streams do not share one video set: {sorted(video_sets)}")
    result = {
        "schema_version": "trackocd.cvpr.score-contract-geometry-smoke-audit.v1",
        "status": "COMPLETE_CATEGORY_FREE_SCORE_CONTRACT_GEOMETRY_SMOKES",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "videos": list(next(iter(video_sets))),
            "iou_threshold": 0.5,
            "official_ocd_or_trackeval_metrics": False,
            "frontend_qualification": False,
            "model_selection": False,
            "mapping_persisted": False,
        },
        "inputs": {
            "geometry_only_gt_index": str(gt_path.resolve()),
            "geometry_only_gt_index_sha256": _sha256(gt_path),
        },
        "mechanisms": records,
        "boundary": {
            "category_fields_read": False,
            "role_or_vocabulary_read": False,
            "val_labels_read": False,
            "test_accessed": False,
            "native_evaluator_rows_read": False,
            "pixels_read": False,
            "feature_arrays_read": False,
            "training_started": False,
            "prediction_to_gt_mapping_persisted": False,
        },
        "interpretation": (
            "This is a paired geometry diagnostic for three score/tracker mechanisms "
            "on the same two anonymous Train videos. It is not an official OCD/MOT "
            "score, a frontend qualification, or Goal A/B/C acceptance."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream", action="append", required=True, metavar="NAME=PATH")
    args = parser.parse_args()
    streams: dict[str, Path] = {}
    for item in args.stream:
        if "=" not in item:
            raise SystemExit("--stream must be NAME=PATH")
        name, raw_path = item.split("=", 1)
        if not name or not raw_path or name in streams:
            raise SystemExit("--stream names must be nonempty and unique")
        streams[name] = Path(raw_path).resolve()
    result = audit(args.geometry_gt.resolve(), streams, args.output.resolve())
    print(json.dumps({"status": result["status"], "mechanisms": result["mechanisms"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
