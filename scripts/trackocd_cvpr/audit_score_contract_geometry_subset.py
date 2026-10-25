#!/usr/bin/env python3
"""Compare a bounded score-contract candidate with a subset of a full stream.

The baseline stream may contain more approved anonymous Train videos than the
candidate.  This audit filters both streams by a predeclared video-ID set
while streaming physical category-free rows; it never opens native evaluator
rows, pixels, semantic fields, Val/Test labels, or feature arrays, and never
persists a prediction-to-GT mapping.  The result is mechanism evidence only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
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
    _stream_frame_bases,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_stats(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(values),
        "finite": len(finite),
        "nonfinite": len(values) - len(finite),
        "min": min(finite) if finite else None,
        "mean": statistics.fmean(finite) if finite else None,
        "max": max(finite) if finite else None,
        "under_0.05": sum(value < 0.05 for value in finite),
        "over_0.5": sum(value > 0.5 for value in finite),
    }


def _source_binding(receipt: dict[str, Any]) -> dict[str, Any]:
    config = Path(str(receipt.get("config", ""))).resolve()
    if not config.is_file() or len(config.parents) < 3:
        raise FileNotFoundError(f"receipt config is not a file: {config}")
    root = config.parents[2]
    bbox_head = root / "ovtrack/models/roi_heads/bbox_heads/convfc_bbox_head.py"
    source = bbox_head.read_text(encoding="utf-8")
    return {
        "config": str(config),
        "config_sha256": _sha256(config),
        "source_root": str(root),
        "bbox_head": str(bbox_head),
        "bbox_head_sha256": _sha256(bbox_head),
        "foreground_mass_candidate_fragment_present": (
            "foreground_scores = scores[:, :-1].sum(dim=-1)" in source
        ),
    }


def _read_stream(
    path: Path,
    expected_videos: set[int],
    gt_by_frame: dict[int, dict[int, list[Any]]],
    gt_track_counts: dict[int, dict[str, int]],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    bases = _stream_frame_bases(path)
    state = _new_stream_state()
    selected_rows = 0
    selected_observations = 0
    scores: list[float] = []
    seen_tracks: set[tuple[int, str]] = set()
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
            if video_id not in expected_videos:
                continue
            if video_id not in gt_track_counts or video_id not in bases:
                raise ValueError(f"stream video {video_id} is outside the approved geometry index")
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
                raise ValueError(f"duplicate physical track {track_key} at {path}:{line_number}")
            seen_tracks.add(track_key)
            state["videos"].add(video_id)
            state["tracks"][track_key] = len(frame_ids)
            for frame_id, box, raw_score in zip(frame_ids, boxes, quality):
                if not _finite_box(box):
                    raise ValueError(f"invalid box at {path}:{line_number}")
                scores.append(float(raw_score))
                frame_index = int(frame_id) - bases[video_id]
                if frame_index in gt_by_frame.get(video_id, {}):
                    state["by_frame"][(video_id, frame_index)].append(
                        (track_key, [float(value) for value in box])
                    )
                    state["annotated_observations"] += 1
            state["rows"] += 1
            state["observations"] += len(frame_ids)
            selected_rows += 1
            selected_observations += len(frame_ids)

    if require_complete and state["videos"] != expected_videos:
        raise ValueError(
            f"selected stream videos {sorted(state['videos'])} != expected {sorted(expected_videos)}"
        )
    geometry = _finalize_stream(state, gt_by_frame, gt_track_counts)
    predicted_on_gt = int(geometry["predicted_observations_on_gt_images"])
    gt_observations = int(geometry["gt_observations"])
    gt_tracks = int(geometry["gt_tracks"])
    geometry.update(
        {
            "observation_precision_on_annotated_images": geometry["per_image_matched_observations"]
            / max(predicted_on_gt, 1),
            "observation_recall_against_gt": geometry["per_image_matched_observations"]
            / max(gt_observations, 1),
            "reliable_gt_track_recall": geometry["reliable_gt_tracks"] / max(gt_tracks, 1),
        }
    )
    return {
        "stream": {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "selected_videos": sorted(state["videos"]),
        },
        "selected_rows": selected_rows,
        "selected_observations": selected_observations,
        "score": _score_stats(scores),
        "geometry": geometry,
    }


def _aggregate_geometry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    matched = totals.get("per_image_matched_observations", 0)
    predicted_on_gt = totals.get("predicted_observations_on_gt_images", 0)
    gt_observations = totals.get("gt_observations", 0)
    gt_tracks = totals.get("gt_tracks", 0)
    totals["observation_precision_on_annotated_images"] = matched / max(predicted_on_gt, 1)
    totals["observation_recall_against_gt"] = matched / max(gt_observations, 1)
    totals["reliable_gt_track_recall"] = totals.get("reliable_gt_tracks", 0) / max(gt_tracks, 1)
    return totals


def _aggregate_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(int(row["count"]) for row in rows)
    finite = sum(int(row["finite"]) for row in rows)
    return {
        "count": count,
        "finite": finite,
        "nonfinite": count - finite,
        "min": min((row["min"] for row in rows if row["min"] is not None), default=None),
        "mean": (
            sum(float(row["mean"]) * int(row["finite"]) for row in rows if row["mean"] is not None)
            / finite
            if finite
            else None
        ),
        "max": max((row["max"] for row in rows if row["max"] is not None), default=None),
        "under_0.05": sum(int(row["under_0.05"]) for row in rows),
        "over_0.5": sum(int(row["over_0.5"]) for row in rows),
    }


def _read_streams(
    paths: list[Path],
    expected_videos: set[int],
    gt_by_frame: dict[int, dict[int, list[Any]]],
    gt_track_counts: dict[int, dict[str, int]],
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one physical stream is required")
    parts = [
        _read_stream(path, expected_videos, gt_by_frame, gt_track_counts, require_complete=False)
        for path in paths
    ]
    observed: set[int] = set()
    for part in parts:
        part_videos = set(part["stream"]["selected_videos"])
        overlap = observed & part_videos
        if overlap:
            raise ValueError(f"video IDs appear in multiple input streams: {sorted(overlap)}")
        observed.update(part_videos)
    if observed != expected_videos:
        raise ValueError(
            f"selected stream videos {sorted(observed)} != expected {sorted(expected_videos)}"
        )
    return {
        "stream": {
            "paths": [part["stream"] for part in parts],
            "selected_videos": sorted(observed),
        },
        "selected_rows": sum(int(part["selected_rows"]) for part in parts),
        "selected_observations": sum(int(part["selected_observations"]) for part in parts),
        "score": _aggregate_score([part["score"] for part in parts]),
        "geometry": _aggregate_geometry([part["geometry"] for part in parts]),
    }


def audit(
    *,
    geometry_gt: Path,
    baseline_streams: list[Path],
    candidate_streams: list[Path],
    baseline_receipt: Path,
    candidate_receipt: Path,
    video_ids: list[int],
    output: Path,
    candidate_mechanism: str,
) -> dict[str, Any]:
    expected_videos = set(video_ids)
    if not expected_videos or len(expected_videos) != len(video_ids):
        raise ValueError("video IDs must be a non-empty unique set")
    gt_by_frame, gt_track_counts = _load_geometry_gt(geometry_gt)
    baseline = _read_streams(baseline_streams, expected_videos, gt_by_frame, gt_track_counts)
    candidate = _read_streams(candidate_streams, expected_videos, gt_by_frame, gt_track_counts)
    baseline_receipt_data = json.loads(baseline_receipt.read_text(encoding="utf-8"))
    candidate_receipt_data = json.loads(candidate_receipt.read_text(encoding="utf-8"))
    result = {
        "schema_version": "trackocd.cvpr.score-contract-geometry-subset-audit.v1",
        "status": "COMPLETE_CATEGORY_FREE_SCORE_CONTRACT_GEOMETRY_SUBSET",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "videos": sorted(expected_videos),
            "iou_threshold": 0.5,
            "official_ocd_or_trackeval_metrics": False,
            "frontend_qualification": False,
            "model_selection": False,
        },
        "inputs": {
            "geometry_only_gt_index": str(geometry_gt.resolve()),
            "geometry_only_gt_index_sha256": _sha256(geometry_gt),
            "baseline_streams": [str(path.resolve()) for path in baseline_streams],
            "candidate_streams": [str(path.resolve()) for path in candidate_streams],
            "baseline_receipt": str(baseline_receipt.resolve()),
            "candidate_receipt": str(candidate_receipt.resolve()),
        },
        "mechanisms": {
            "baseline": "tracked_double_softmax_max_class",
            "candidate": candidate_mechanism,
        },
        "baseline": baseline,
        "candidate": candidate,
        "deltas_candidate_minus_baseline": {
            "selected_rows": candidate["selected_rows"] - baseline["selected_rows"],
            "selected_observations": candidate["selected_observations"] - baseline["selected_observations"],
            "score_mean": candidate["score"]["mean"] - baseline["score"]["mean"],
            "geometry_precision": (
                candidate["geometry"]["observation_precision_on_annotated_images"]
                - baseline["geometry"]["observation_precision_on_annotated_images"]
            ),
            "geometry_recall": (
                candidate["geometry"]["observation_recall_against_gt"]
                - baseline["geometry"]["observation_recall_against_gt"]
            ),
        },
        "native_output_identity": {
            "baseline_bbox_sha256": baseline_receipt_data.get("native_bbox_output_sha256"),
            "candidate_bbox_sha256": candidate_receipt_data.get("native_bbox_output_sha256"),
            "bbox_byte_identical": baseline_receipt_data.get("native_bbox_output_sha256")
            == candidate_receipt_data.get("native_bbox_output_sha256"),
            "baseline_track_sha256": baseline_receipt_data.get("native_track_output_sha256"),
            "candidate_track_sha256": candidate_receipt_data.get("native_track_output_sha256"),
            "track_byte_identical": baseline_receipt_data.get("native_track_output_sha256")
            == candidate_receipt_data.get("native_track_output_sha256"),
        },
        "source_binding": {
            "baseline": _source_binding(baseline_receipt_data),
            "candidate": _source_binding(candidate_receipt_data),
        },
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
            "Paired category-free geometry evidence on the same fixed anonymous Train videos. "
            "All input streams are read sequentially and filtered in memory; "
            "no filtered prediction file or GT mapping is persisted. This is mechanism evidence, "
            "not an official OCD/MOT score, frontend qualification, or Goal A/B/C acceptance."
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
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--video-id", type=int, action="append", required=True)
    parser.add_argument("--candidate-mechanism", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        geometry_gt=args.geometry_gt.resolve(),
        baseline_streams=[path.resolve() for path in args.baseline_stream],
        candidate_streams=[path.resolve() for path in args.candidate_stream],
        baseline_receipt=args.baseline_receipt.resolve(),
        candidate_receipt=args.candidate_receipt.resolve(),
        video_ids=args.video_id,
        output=args.output.resolve(),
        candidate_mechanism=args.candidate_mechanism,
    )
    print(json.dumps({
        "status": result["status"],
        "scope": result["scope"],
        "baseline": result["baseline"],
        "candidate": result["candidate"],
        "deltas_candidate_minus_baseline": result["deltas_candidate_minus_baseline"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
