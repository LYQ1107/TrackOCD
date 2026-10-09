#!/usr/bin/env python3
"""Audit the public predicted track stream without visual features or GT.

The audit is deliberately a single sequential pass over the JSONL stream.
It records the raw stream lineage, observation/box integrity, duration shape,
score distributions, and evidence needed to distinguish final physical tracks
from raw or fragmented tracklets.  No image is opened and no evaluator label
is loaded.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


SOURCE = ROOT / "data/tao_ow_ocd_v1/public/pred_track_stream.jsonl"
OUTPUT = OUTPUT_TARGET / "audit/predicted_stream_statistics.json"
V2_REGISTRATION = OUTPUT_TARGET / "audit/predicted_track_stream.json"
V2_BUILDER = ROOT / "scripts/trackocd_v2/build_predicted_stream.py"
UPSTREAM_BUILDER = ROOT / "scripts/merge_simowt_output.py"
UPSTREAM_PREDICTIONS = ROOT / "outputs/simowt/val_predictions.json"


def _metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "bytes": int(stat.st_size),
        "mtime": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


def _quantiles(values: Iterable[float]) -> dict[str, Any]:
    finite = np.asarray(list(values), dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "quantiles": {}}
    quantile_values = np.quantile(finite, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "quantiles": {
            name: float(value)
            for name, value in zip(("p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"), quantile_values)
        },
    }


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {"count": int(numerator), "denominator": int(denominator), "ratio": float(numerator / denominator) if denominator else None}


def _score_bins(values: Iterable[float]) -> dict[str, int]:
    bins = Counter({
        "<0": 0,
        "0-0.25": 0,
        "0.25-0.50": 0,
        "0.50-0.75": 0,
        "0.75-1": 0,
        ">=1": 0,
    })
    for value in values:
        value = float(value)
        if value < 0:
            key = "<0"
        elif value < 0.25:
            key = "0-0.25"
        elif value < 0.50:
            key = "0.25-0.50"
        elif value < 0.75:
            key = "0.50-0.75"
        elif value < 1.0:
            key = "0.75-1"
        else:
            key = ">=1"
        bins[key] += 1
    return dict(bins)


def _video_distribution(counts: dict[str, int]) -> dict[str, Any]:
    values = np.asarray(list(counts.values()), dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "quantiles": {}, "counts": {}}
    q = np.quantile(values, [0.75, 0.90, 0.95, 0.99])
    return {
        "count": int(values.size),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "quantiles": {name: float(value) for name, value in zip(("p75", "p90", "p95", "p99"), q)},
        "counts": {str(key): int(value) for key, value in sorted(counts.items(), key=lambda item: str(item[0]))},
    }


def _track_length_bucket(length: int) -> str:
    if length == 1:
        return "1"
    if length == 2:
        return "2"
    if 3 <= length <= 4:
        return "3-4"
    if 5 <= length <= 8:
        return "5-8"
    if 9 <= length <= 16:
        return "9-16"
    if length > 16:
        return ">16"
    return "0"


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def main() -> int:
    out = ensure_output_layout()
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    track_count = 0
    observation_count = 0
    video_track_counts: Counter[str] = Counter()
    exact_lengths: Counter[int] = Counter()
    bucket_lengths: Counter[str] = Counter()
    track_id_to_videos: defaultdict[str, set[str]] = defaultdict(set)
    video_track_keys: set[tuple[str, str]] = set()
    sample_keys: set[str] = set()
    score_values = array("d")
    track_mean_scores = array("d")

    counters: Counter[str] = Counter()
    examples: defaultdict[str, list[str]] = defaultdict(list)
    previous_stream_order: int | None = None

    def note(name: str, example: str | None = None) -> None:
        counters[name] += 1
        if example is not None and len(examples[name]) < 5:
            examples[name].append(example)

    with SOURCE.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                note("blank_lines")
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                note("json_parse_error_rows", f"line {line_number}: {exc}")
                continue
            if not isinstance(row, dict):
                note("non_object_rows", f"line {line_number}")
                continue

            track_count += 1
            sample_id = str(row.get("sample_id", ""))
            video_id = str(row.get("video_id", ""))
            track_id = str(row.get("track_id", ""))
            key = (video_id, track_id)
            if key in video_track_keys:
                note("duplicate_video_track_keys", f"{video_id}/{track_id}")
            video_track_keys.add(key)
            track_id_to_videos[track_id].add(video_id)
            video_track_counts[video_id] += 1

            if sample_id in sample_keys:
                note("duplicate_sample_ids", sample_id)
            sample_keys.add(sample_id)

            stream_order = _float_or_none(row.get("stream_order"))
            if stream_order is None or not stream_order.is_integer():
                note("invalid_stream_order", sample_id)
            else:
                stream_order_int = int(stream_order)
                if previous_stream_order is not None:
                    if stream_order_int == previous_stream_order:
                        note("duplicate_stream_orders", str(stream_order_int))
                    elif stream_order_int < previous_stream_order:
                        note("stream_order_regressions", f"{stream_order_int} after {previous_stream_order}")
                previous_stream_order = stream_order_int

            frame_ids = row.get("frame_ids")
            boxes = row.get("boxes_xyxy")
            image_paths = row.get("image_paths")
            scores = row.get("scores")
            areas = row.get("areas")
            if not isinstance(frame_ids, list):
                note("missing_or_nonlist_frame_ids", sample_id)
                frame_ids = []
            if not isinstance(boxes, list):
                note("missing_or_nonlist_boxes", sample_id)
                boxes = []
            if not isinstance(image_paths, list):
                note("missing_or_nonlist_image_paths", sample_id)
                image_paths = []
            if not isinstance(scores, list):
                note("missing_or_nonlist_scores", sample_id)
                scores = []
            if not isinstance(areas, list):
                note("missing_or_nonlist_areas", sample_id)
                areas = []

            length = len(frame_ids)
            if length == 0:
                note("empty_tracks", sample_id)
            exact_lengths[length] += 1
            bucket_lengths[_track_length_bucket(length)] += 1
            observation_count += length

            if len({len(frame_ids), len(boxes), len(image_paths), len(scores), len(areas)}) != 1:
                note("lineage_length_mismatch_rows", sample_id)
            if len(boxes) != length:
                note("box_observation_count_mismatch_rows", sample_id)
            if len(scores) != length:
                note("score_observation_count_mismatch_rows", sample_id)
            if len(image_paths) != length:
                note("image_path_observation_count_mismatch_rows", sample_id)

            try:
                if len(frame_ids) != len(set(frame_ids)):
                    note("duplicate_frame_ids_within_track", sample_id)
            except TypeError:
                note("unhashable_frame_ids", sample_id)
            if any(isinstance(frame_ids[i], (int, float)) and isinstance(frame_ids[i + 1], (int, float)) and frame_ids[i + 1] < frame_ids[i] for i in range(max(0, length - 1))):
                note("nonmonotonic_frame_ids", sample_id)

            finite_track_scores: list[float] = []
            for obs_index in range(length):
                if obs_index >= len(boxes):
                    note("missing_box_observations", f"{sample_id}[{obs_index}]")
                else:
                    box = boxes[obs_index]
                    if not isinstance(box, (list, tuple)) or len(box) != 4:
                        note("invalid_box_count", f"{sample_id}[{obs_index}]")
                    else:
                        coords = [_float_or_none(value) for value in box]
                        if any(value is None or not math.isfinite(value) for value in coords):
                            note("invalid_box_count", f"{sample_id}[{obs_index}]")
                        else:
                            x1, y1, x2, y2 = (float(value) for value in coords)  # type: ignore[arg-type]
                            if min(x1, y1, x2, y2) < 0:
                                note("negative_coordinate_boxes", f"{sample_id}[{obs_index}]")
                            if x2 <= x1 or y2 <= y1:
                                note("degenerate_box_count", f"{sample_id}[{obs_index}]")
                            source_area = _float_or_none(areas[obs_index]) if obs_index < len(areas) else None
                            geometric_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                            if source_area is None:
                                note("invalid_area_count", f"{sample_id}[{obs_index}]")
                            elif math.isfinite(source_area) and abs(source_area - geometric_area) > max(1e-3, 1e-6 * max(1.0, geometric_area)):
                                note("area_field_mismatch_count", f"{sample_id}[{obs_index}]")
                if obs_index >= len(scores):
                    note("missing_score_observations", f"{sample_id}[{obs_index}]")
                else:
                    score = _float_or_none(scores[obs_index])
                    if score is None or not math.isfinite(score):
                        note("invalid_native_score_count", f"{sample_id}[{obs_index}]")
                    else:
                        score_values.append(score)
                        finite_track_scores.append(score)
                        if score < 0.0 or score > 1.0:
                            note("out_of_range_native_score_count", f"{sample_id}[{obs_index}]")
            if finite_track_scores:
                track_mean_scores.append(float(np.mean(finite_track_scores)))

    track_id_reused_across_videos = {track_id: sorted(videos) for track_id, videos in track_id_to_videos.items() if len(videos) > 1}
    exact_histogram = {str(length): int(count) for length, count in sorted(exact_lengths.items())}
    required_buckets = {name: int(bucket_lengths.get(name, 0)) for name in ("1", "2", "3-4", "5-8", "9-16", ">16")}
    length_values = np.repeat(
        np.asarray(list(exact_lengths.keys()), dtype=np.float64),
        np.asarray(list(exact_lengths.values()), dtype=np.int64),
    )
    length_quantiles = np.quantile(length_values, [0.50, 0.75, 0.90, 0.95, 0.99]) if length_values.size else []
    registration = {}
    if V2_REGISTRATION.exists():
        try:
            registration = json.loads(V2_REGISTRATION.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            registration = {"status": "UNPARSEABLE"}

    errors = {name: int(value) for name, value in sorted(counters.items())}
    total_tracks = track_count
    result = {
        "schema_version": "trackocd.v2.predicted_stream_statistics.v1",
        "status": "COMPLETE" if not errors.get("json_parse_error_rows") and not errors.get("non_object_rows") else "COMPLETE_WITH_SOURCE_ERRORS",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "audit_contract": {
            "features_read": False,
            "images_opened": False,
            "gt_join_used": False,
            "test_semantic_accessed": False,
            "model_decisions_produced": False,
        },
        "stream_totals": {
            "total_tracks": int(total_tracks),
            "total_observations": int(observation_count),
            "videos": int(len(video_track_counts)),
            "tracks_per_video": _video_distribution(dict(video_track_counts)),
        },
        "track_length": {
            "exact_histogram": exact_histogram,
            "required_histogram": required_buckets,
            "summary": {
                "mean": float(np.mean(length_values)) if length_values.size else None,
                "median": float(np.median(length_values)) if length_values.size else None,
                "max": int(np.max(length_values)) if length_values.size else None,
                "quantiles": {
                    name: float(value)
                    for name, value in zip(("p50", "p75", "p90", "p95", "p99"), length_quantiles)
                },
            },
            "single_frame_ratio": _ratio(exact_lengths.get(1, 0), total_tracks),
            "less_than_4_observation_ratio": _ratio(sum(count for length, count in exact_lengths.items() if length < 4), total_tracks),
            "less_than_8_observation_ratio": _ratio(sum(count for length, count in exact_lengths.items() if length < 8), total_tracks),
            "less_than_16_observation_ratio": _ratio(sum(count for length, count in exact_lengths.items() if length < 16), total_tracks),
        },
        "native_score_distribution": {
            "observation_scores": _quantiles(score_values),
            "observation_score_bins": _score_bins(score_values),
            "track_mean_scores": _quantiles(track_mean_scores),
            "track_mean_score_bins": _score_bins(track_mean_scores),
            "invalid_score_observations": int(counters.get("invalid_native_score_count", 0)),
            "out_of_range_score_observations": int(counters.get("out_of_range_native_score_count", 0)),
        },
        "box_integrity": {
            "invalid_box_count": int(counters.get("invalid_box_count", 0)),
            "degenerate_box_count": int(counters.get("degenerate_box_count", 0)),
            "negative_coordinate_box_count": int(counters.get("negative_coordinate_boxes", 0)),
            "invalid_area_count": int(counters.get("invalid_area_count", 0)),
            "area_field_mismatch_count": int(counters.get("area_field_mismatch_count", 0)),
        },
        "lineage_integrity": {
            "schema_or_lineage_error_counts": errors,
            "error_examples": {name: values for name, values in sorted(examples.items())},
            "duplicate_video_track_keys": int(counters.get("duplicate_video_track_keys", 0)),
            "duplicate_sample_ids": int(counters.get("duplicate_sample_ids", 0)),
            "duplicate_frame_ids_within_track": int(counters.get("duplicate_frame_ids_within_track", 0)),
            "nonmonotonic_frame_ids": int(counters.get("nonmonotonic_frame_ids", 0)),
            "stream_order_regressions": int(counters.get("stream_order_regressions", 0)),
            "duplicate_stream_orders": int(counters.get("duplicate_stream_orders", 0)),
            "track_ids_reused_across_videos": int(len(track_id_reused_across_videos)),
            "track_id_reuse_examples": dict(list(sorted(track_id_reused_across_videos.items()))[:20]),
        },
        "provenance": {
            "source_file": _metadata(SOURCE),
            "v2_registration_script": _metadata(V2_BUILDER),
            "upstream_generation_script": _metadata(UPSTREAM_BUILDER),
            "upstream_prediction_artifact": _metadata(UPSTREAM_PREDICTIONS),
            "v2_registration_audit": registration,
            "source_role": "public SimOWT/Q0 predicted validation stream",
            "source_fields_observed": ["sample_id", "video_id", "track_id", "frame_ids", "image_paths", "boxes_xyxy", "areas", "scores", "stream_order"],
            "filtering_protocol": {
                "upstream_deduplication": "deduplicate per (image_id, track_id) before grouping",
                "grouping_key": "(video_id, track_id)",
                "ordering": "within-track source image/frame order; tracks sorted by video_id and assigned contiguous stream_order",
                "additional_v2_filtering": "none; build_predicted_stream.py only converts/registers the public stream",
                "category_or_distractor_filtering": False,
                "gt_matching_or_category_join": False,
                "score_threshold_applied_by_registration": False,
                "historical_iou_matched_stream_used": False,
            },
        },
        "physical_track_interpretation": {
            "classification": "RAW_OR_FRAGMENTED_TRACKLETS",
            "final_physical_tracks_verified": False,
            "basis": [
                "The stream carries detector/tracker track_id grouped only within each video; the same track_id is reused across videos.",
                "The duration distribution is dominated by very short units; this is inconsistent with treating every row as a verified final physical identity.",
                "No cross-video identity linkage or physical-track lifecycle certificate is present in the public row schema.",
            ],
            "required_next_step": "Use this stream for physical frontend bake-off and report tracklet fragmentation; do not call rows final physical tracks without a separate physical identity audit.",
        },
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
