#!/usr/bin/env python3
"""Full-Train offline score-gate counterfactual for COVTrack-native streams.

This is an evaluator-side, category-free audit.  It reuses only the already
normalized anonymous physical streams and the sealed geometry-only physical GT
index.  It does not rerun COVTrack: each threshold is an explicit counterfactual
retention rule over the exported per-observation ``quality`` field.  The result
therefore diagnoses score/lifecycle sensitivity, but is not a replacement for a
fresh native detector/tracker run or a formal OCD/MOT metric.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trackocd_cvpr.audit_predicted_train_geometry import (
    DEFAULT_GT,
    DEFAULT_RECONCILIATION,
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

DEFAULT_CONFIG = ROOT / "configs/trackocd_cvpr/covtrack_score_gate_full_train_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_score_gate_full_train_v1/result.json"
IOU_THRESHOLD = 0.5


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _threshold_key(value: float) -> str:
    return f"{value:.16g}"


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "trackocd.cvpr.covtrack_score_gate_full_train.v1":
        raise ValueError("unexpected score-gate config schema")
    if payload.get("status") != "FROZEN_OFFLINE_SCORE_GATE_COUNTERFACTUAL":
        raise ValueError("score-gate config is not frozen")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("score-gate config has no thresholds")
    values = [float(item) for item in thresholds]
    if values != sorted(set(values)) or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("score-gate thresholds are not sorted unique probabilities")
    if payload.get("geometry_matching") != "per-image class-free Hungarian matching at IoU >= 0.5 against the sealed physical geometry-only Train index":
        raise ValueError("unexpected score-gate geometry contract")
    if payload.get("boundaries", {}).get("test_accessed") is not False:
        raise ValueError("score-gate config does not close Test")
    return payload


def _validate_row(row: dict[str, Any], path: Path, line_number: int) -> None:
    if set(row) != PHYSICAL_FIELDS:
        raise ValueError(f"physical stream schema mismatch at {path}:{line_number}")
    if FORBIDDEN_FIELDS & set(row):
        raise ValueError(f"semantic/evaluator field found at {path}:{line_number}")
    if row.get("source_split") != "train_predicted":
        raise ValueError(f"unexpected source split at {path}:{line_number}")


def _new_stream_state() -> dict[str, Any]:
    return {
        "tracks": {},
        "by_frame": defaultdict(list),
        "videos": set(),
        "rows": 0,
        "observations": 0,
        "annotated_observations": 0,
        "score_counts": Counter(),
        "score_sum": 0.0,
        "score_min": math.inf,
        "score_max": -math.inf,
        "score_invalid": 0,
        "score_out_of_range": 0,
    }


def _read_stream(
    path: Path,
    gt_by_frame: dict[int, dict[int, list[tuple[str, list[float]]]]],
    gt_track_counts: dict[int, dict[str, int]],
    thresholds: list[float],
    seen_videos: set[int],
    seen_tracks: set[tuple[int, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _new_stream_state()
    for threshold in thresholds:
        key = _threshold_key(threshold)
        state["score_counts"][f"observations_ge_{key}"] = 0
    frame_bases = _stream_frame_bases(path)
    digest = hashlib.sha256()
    first_video = last_video = None
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            _validate_row(row, path, line_number)
            video_id = int(row["video_id"])
            if video_id not in gt_track_counts:
                raise ValueError(f"stream video {video_id} is outside the approved geometry index")
            if video_id not in state["videos"]:
                if video_id in seen_videos:
                    raise ValueError(f"video {video_id} appears in multiple stream positions")
                state["videos"].add(video_id)
                seen_videos.add(video_id)
            first_video = video_id if first_video is None else first_video
            last_video = video_id

            track_id = str(row["physical_track_id"])
            track_key = (video_id, track_id)
            if track_key in seen_tracks or track_key in state["tracks"]:
                raise ValueError(f"duplicate predicted physical track {track_key}")
            seen_tracks.add(track_key)
            frame_ids = row["frame_ids"]
            boxes = row["boxes_xyxy"]
            image_paths = row["image_paths"]
            quality = row["quality"]
            if not all(isinstance(value, list) for value in (frame_ids, boxes, image_paths, quality)):
                raise ValueError(f"non-list physical observation field at {path}:{line_number}")
            if not (len(frame_ids) == len(boxes) == len(image_paths) == len(quality)) or not frame_ids:
                raise ValueError(f"physical observation lengths disagree at {path}:{line_number}")
            if len(set(frame_ids)) != len(frame_ids):
                raise ValueError(f"duplicate predicted frame in {track_key}")

            track_max = -math.inf
            for frame_id, box, score in zip(frame_ids, boxes, quality):
                if not _finite_box(box):
                    raise ValueError(f"invalid predicted box at {path}:{line_number}")
                state["observations"] += 1
                state["score_counts"]["total"] += 1
                try:
                    numeric_score = float(score)
                except (TypeError, ValueError):
                    numeric_score = math.nan
                if not math.isfinite(numeric_score):
                    state["score_invalid"] += 1
                    continue
                if numeric_score < 0.0 or numeric_score > 1.0:
                    state["score_out_of_range"] += 1
                    continue
                state["score_sum"] += numeric_score
                state["score_min"] = min(state["score_min"], numeric_score)
                state["score_max"] = max(state["score_max"], numeric_score)
                track_max = max(track_max, numeric_score)
                for threshold in thresholds:
                    if numeric_score >= threshold:
                        state["score_counts"][f"observations_ge_{_threshold_key(threshold)}"] += 1
                frame_index = int(frame_id) - frame_bases[video_id]
                if frame_index < 0:
                    raise ValueError(f"predicted frame ID precedes video base at {path}:{line_number}")
                if frame_index in gt_by_frame.get(video_id, {}):
                    state["annotated_observations"] += 1
                    state["by_frame"][(video_id, frame_index)].append(
                        (track_key, [float(value) for value in box], numeric_score)
                    )
            state["tracks"][track_key] = track_max
            state["rows"] += 1

    for predictions in state["by_frame"].values():
        for _track_key, _box, score in predictions:
            for threshold in thresholds:
                key = _threshold_key(threshold)
                if score >= threshold:
                    state["score_counts"][f"annotated_observations_ge_{key}"] += 1
    # The per-track maximum is sufficient to count retained physical tracks;
    # observation counts are recomputed from the frame records and the total
    # stream count for non-annotated frames in the pass below.
    stream_record = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "rows": state["rows"],
        "observations": state["observations"],
        "videos": len(state["videos"]),
        "first_video_id": first_video,
        "last_video_id": last_video,
    }
    return state, stream_record


def _evaluate_threshold(
    states: list[dict[str, Any]],
    gt_by_frame: dict[int, dict[int, list[tuple[str, list[float]]]]],
    gt_track_counts: dict[int, dict[str, int]],
    gt_track_frames: dict[tuple[int, str], list[int]],
    threshold: float,
) -> dict[str, int | float]:
    totals: Counter[str] = Counter()
    retained_track_keys: set[tuple[int, str]] = set()
    matched_pred: set[tuple[int, str]] = set()
    matched_gt: set[tuple[int, str]] = set()
    matched_by_gt: dict[tuple[int, str], list[tuple[int, tuple[int, str]]]] = defaultdict(list)
    for state in states:
        for track_key, max_score in state["tracks"].items():
            if max_score >= threshold:
                retained_track_keys.add(track_key)
        key = _threshold_key(threshold)
        totals["stream_rows"] += state["rows"]
        totals["stream_observations"] += state["observations"]
        totals["retained_annotated_observations"] += sum(
            1 for predictions in state["by_frame"].values() for _track_key, _box, score in predictions if score >= threshold
        )
        # Counts below are derived from every physical observation, including
        # frames outside the geometry index, without retaining those frames.
        # A second compact count is built from the stream-level quality pass.
        totals[f"retained_observations_{key}"] += state["score_counts"].get(f"observations_ge_{key}", 0)
        for (video_id, frame_index), predictions in state["by_frame"].items():
            selected = [(track_key, box) for track_key, box, score in predictions if score >= threshold]
            if not selected:
                continue
            gt_rows = gt_by_frame.get(video_id, {}).get(frame_index, [])
            if not gt_rows:
                continue
            gt_keys = [(video_id, gt_key) for gt_key, _box in gt_rows]
            gt_boxes = [box for _gt_key, box in gt_rows]
            pred_keys = [track_key for track_key, _box in selected]
            pred_boxes = [box for _track_key, box in selected]
            matrix = _box_iou_matrix(gt_boxes, pred_boxes)
            bonus = max(matrix.shape) + 1
            weights = np.where(matrix >= IOU_THRESHOLD, bonus + matrix, 0.0)
            gt_indices, pred_indices = linear_sum_assignment(weights, maximize=True)
            for gt_index, pred_index in zip(gt_indices, pred_indices):
                if weights[gt_index, pred_index] <= 0:
                    continue
                totals["per_image_matched_observations"] += 1
                pred_key = pred_keys[int(pred_index)]
                gt_key = gt_keys[int(gt_index)]
                matched_pred.add(pred_key)
                matched_gt.add(gt_key)
                matched_by_gt[gt_key].append((frame_index, pred_key))
    totals["videos"] = sum(len(state["videos"]) for state in states)
    totals["predicted_tracks"] = len(retained_track_keys)
    totals["gt_tracks"] = sum(len(gt_track_counts[video_id]) for state in states for video_id in state["videos"])
    totals["gt_observations"] = sum(
        sum(gt_track_counts[video_id].values()) for state in states for video_id in state["videos"]
    )
    totals["predicted_tracks_with_per_image_match"] = len(matched_pred)
    totals["gt_tracks_with_per_image_match"] = len(matched_gt)
    temporal_pairs = temporal_same_id = temporal_switches = 0
    temporal_gt_tracks = set()
    temporal_switch_gt_tracks = set()
    temporal_same_gt_tracks = set()
    for gt_key, frame_indices in gt_track_frames.items():
        matched_at_frame = {frame_index: pred_key for frame_index, pred_key in matched_by_gt.get(gt_key, [])}
        for previous_frame, current_frame in zip(frame_indices, frame_indices[1:]):
            previous_pred = matched_at_frame.get(previous_frame)
            current_pred = matched_at_frame.get(current_frame)
            if previous_pred is None or current_pred is None:
                continue
            temporal_pairs += 1
            temporal_gt_tracks.add(gt_key)
            if previous_pred == current_pred:
                temporal_same_id += 1
                temporal_same_gt_tracks.add(gt_key)
            else:
                temporal_switches += 1
                temporal_switch_gt_tracks.add(gt_key)
    totals["temporal_pairs_both_matched"] = temporal_pairs
    totals["temporal_same_id_pairs"] = temporal_same_id
    totals["temporal_switch_pairs"] = temporal_switches
    totals["temporal_switch_rate_conditional"] = temporal_switches / max(temporal_pairs, 1)
    totals["gt_tracks_with_temporal_pair"] = len(temporal_gt_tracks)
    totals["gt_tracks_with_temporal_switch"] = len(temporal_switch_gt_tracks)
    totals["gt_tracks_with_temporal_same_id"] = len(temporal_same_gt_tracks)
    totals["observation_precision_on_retained_annotated_images"] = totals["per_image_matched_observations"] / max(totals["retained_annotated_observations"], 1)
    totals["observation_recall_against_annotated_gt"] = totals["per_image_matched_observations"] / max(totals["gt_observations"], 1)
    totals["gt_track_recall_with_per_image_match"] = totals["gt_tracks_with_per_image_match"] / max(totals["gt_tracks"], 1)
    return dict(totals)


def audit(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    thresholds = [float(item) for item in config["thresholds"]]
    gt_path = ROOT / config["inputs"]["geometry_gt_index"]
    stream_root = ROOT / config["inputs"]["stream_root"]
    reconciliation_path = ROOT / config["inputs"]["reconciliation"]
    gt_by_frame, gt_track_counts = _load_geometry_gt(gt_path)
    gt_track_frames: dict[tuple[int, str], list[int]] = defaultdict(list)
    for video_id, frames in gt_by_frame.items():
        for frame_index, rows in frames.items():
            for track_id, _box in rows:
                gt_track_frames[(video_id, track_id)].append(frame_index)
    gt_track_frames = {key: sorted(values) for key, values in gt_track_frames.items()}
    paths = _stream_paths(stream_root)
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    if reconciliation.get("status") != "PASS_ALL_407_FIXED_ANONYMOUS_TRAIN_DENSE_SUBSHARDS":
        raise ValueError("full anonymous Train cache reconciliation is not PASS")

    states: list[dict[str, Any]] = []
    stream_records: list[dict[str, Any]] = []
    seen_videos: set[int] = set()
    seen_tracks: set[tuple[int, str]] = set()
    score_total = score_invalid = score_out_of_range = 0
    score_sum = 0.0
    score_min = math.inf
    score_max = -math.inf
    total_rows = total_observations = 0
    for path in paths:
        state, stream_record = _read_stream(path, gt_by_frame, gt_track_counts, thresholds, seen_videos, seen_tracks)
        states.append(state)
        stream_records.append(stream_record)
        total_rows += state["rows"]
        total_observations += state["observations"]
        score_total += state["score_counts"]["total"]
        score_invalid += state["score_invalid"]
        score_out_of_range += state["score_out_of_range"]
        score_sum += state["score_sum"]
        score_min = min(score_min, state["score_min"])
        score_max = max(score_max, state["score_max"])

    expected = reconciliation.get("totals", {})
    if total_rows != int(expected.get("tracks", -1)) or total_observations != int(expected.get("observations", -1)):
        raise ValueError("stream totals disagree with the reconciled full cache")
    if len(seen_videos) != 407 or seen_videos != set(gt_track_counts):
        raise ValueError("full stream video coverage does not equal the approved geometry index")

    threshold_results = {}
    for threshold in thresholds:
        threshold_results[_threshold_key(threshold)] = _evaluate_threshold(
            states, gt_by_frame, gt_track_counts, gt_track_frames, threshold
        )

    result = {
        "schema_version": "trackocd.cvpr.category-free-covtrack-score-gate-full-train.v1",
        "status": "COMPLETE_CATEGORY_FREE_FULL_TRAIN_SCORE_GATE_COUNTERFACTUAL",
        "generated_utc": _now(),
        "config": {
            "path": str(config_path.resolve()),
            "sha256": _sha256(config_path),
            "thresholds": thresholds,
        },
        "inputs": {
            "geometry_only_gt_index": str(gt_path.resolve()),
            "geometry_only_gt_index_sha256": _sha256(gt_path),
            "full_cache_reconciliation": str(reconciliation_path.resolve()),
            "full_cache_reconciliation_sha256": _sha256(reconciliation_path),
            "physical_streams": stream_records,
        },
        "counts": {
            "videos": len(seen_videos),
            "predicted_stream_rows": total_rows,
            "predicted_stream_observations": total_observations,
            "geometry_gt_tracks": sum(len(value) for value in gt_track_counts.values()),
            "geometry_gt_observations": sum(sum(value.values()) for value in gt_track_counts.values()),
        },
        "score_distribution": {
            "finite_in_range_observations": score_total - score_invalid - score_out_of_range,
            "nonfinite_observations": score_invalid,
            "out_of_range_observations": score_out_of_range,
            "min": None if score_total == score_invalid + score_out_of_range else score_min,
            "max": None if score_total == score_invalid + score_out_of_range else score_max,
            "mean": None if score_total == score_invalid + score_out_of_range else score_sum / max(score_total - score_invalid - score_out_of_range, 1),
        },
        "threshold_results": threshold_results,
        "boundary": config["boundaries"],
        "interpretation": "Offline retention counterfactual over an existing anonymous COVTrack-native Train stream; not a fresh detector/tracker run, not formal OCD/MOT, not model selection, and not Goal A/B/C acceptance.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.config.resolve(), args.output.resolve())
    print(json.dumps({
        "status": result["status"],
        "counts": result["counts"],
        "score_distribution": result["score_distribution"],
        "threshold_results": result["threshold_results"],
        "boundary": result["boundary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
