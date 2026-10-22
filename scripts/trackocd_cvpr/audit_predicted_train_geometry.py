#!/usr/bin/env python3
"""Audit category-free geometry overlap between the fixed Train stream and GT.

This is an evaluator-side audit only.  It reads the previously authorized
geometry-only physical GT index and the anonymous COVTrack-native Train
streams.  Category IDs, role labels, vocabulary, pixels, Val/Test data, and
feature arrays are never opened.  The output contains aggregate counts only;
it deliberately does not persist a prediction-to-GT track mapping.

The primary matching unit is per-image class-free IoU Hungarian matching at
IoU >= 0.5.  Track-level temporal-IoU values are formed only from those
geometry observations and are reported as a diagnostic, not as TrackEval or
formal OCD/MOT metrics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GT = ROOT / "outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v10_physical_GT/private_exact_joint_proof_or_assignment.json"
DEFAULT_STREAM_ROOT = ROOT / "outputs/trackocd_v2_runtime/manifests/frontend_streams/covtrack_native/covtrack_train_dense_shards_v1"
DEFAULT_RECONCILIATION = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_shards_v1/full_cache_reconciliation.json"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_train_geometry_audit_v1/result.json"
IOU_THRESHOLD = 0.5
PHYSICAL_FIELDS = {
    "sample_key",
    "source_sample_id",
    "video_id",
    "physical_track_id",
    "frame_ids",
    "boxes_xyxy",
    "image_paths",
    "quality",
    "stream_order",
    "source_split",
}
FORBIDDEN_FIELDS = {
    "category_id",
    "category_ids",
    "category_name",
    "gt_category_id",
    "gt_split",
    "gt_match_id",
    "text",
    "novel",
    "label",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_box(box: Any) -> bool:
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in box)
        and float(box[2]) > float(box[0])
        and float(box[3]) > float(box[1])
    )


def _box_iou_matrix(gt_boxes: list[list[float]], pred_boxes: list[list[float]]) -> np.ndarray:
    gt = np.asarray(gt_boxes, dtype=np.float64)
    pred = np.asarray(pred_boxes, dtype=np.float64)
    intersection = np.maximum(
        0.0,
        np.minimum(gt[:, None, 2:], pred[None, :, 2:])
        - np.maximum(gt[:, None, :2], pred[None, :, :2]),
    ).prod(axis=2)
    gt_area = np.maximum(0.0, gt[:, 2] - gt[:, 0]) * np.maximum(0.0, gt[:, 3] - gt[:, 1])
    pred_area = np.maximum(0.0, pred[:, 2] - pred[:, 0]) * np.maximum(0.0, pred[:, 3] - pred[:, 1])
    union = gt_area[:, None] + pred_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _load_geometry_gt(path: Path) -> tuple[dict[int, dict[int, list[tuple[str, list[float]]]]], dict[int, dict[str, int]]]:
    """Load only the previously sealed geometry-only physical index."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "trackocd.cvpr.private-physical-GT.v1":
        raise ValueError("unexpected physical GT index schema")
    if payload.get("not_model_facing") is not True or payload.get("not_a_frozen_protocol") is not True:
        raise ValueError("physical GT index is not marked evaluator-only")
    if payload.get("semantic_category_labels_retained") is not False:
        raise ValueError("physical GT index retains semantic labels")
    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("physical GT index has no track list")

    by_video_frame: dict[int, dict[int, list[tuple[str, list[float]]]]] = defaultdict(lambda: defaultdict(list))
    by_video_track_counts: dict[int, dict[str, int]] = defaultdict(dict)
    image_video: dict[int, int] = {}
    image_frame: dict[int, int] = {}
    seen_tracks: set[tuple[int, str]] = set()
    for row in tracks:
        if set(row) != {"video_id", "opaque_track_id", "observations"}:
            raise ValueError("physical GT row contains non-geometry fields")
        video_id = row["video_id"]
        track_id = row["opaque_track_id"]
        if type(video_id) is not int or type(track_id) is not int or video_id < 0 or track_id < 0:
            raise ValueError("invalid opaque physical GT key")
        track_key = str(track_id)
        if (video_id, track_key) in seen_tracks:
            raise ValueError("duplicate physical GT track")
        seen_tracks.add((video_id, track_key))
        observations = row["observations"]
        if not isinstance(observations, list) or not observations:
            raise ValueError("empty physical GT track")
        count = 0
        last_frame = -1
        for observation in observations:
            if not isinstance(observation, list) or len(observation) != 6:
                raise ValueError("physical GT observation is not geometry-only [image,frame,xywh]")
            image_id, frame_index = observation[:2]
            if type(image_id) is not int or type(frame_index) is not int or image_id < 0 or frame_index < 0:
                raise ValueError("invalid physical GT image/frame key")
            if frame_index <= last_frame:
                raise ValueError("physical GT track chronology is not strict")
            last_frame = frame_index
            x, y, width, height = (float(value) for value in observation[2:])
            if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
                raise ValueError("invalid physical GT xywh box")
            if image_id in image_video and image_video[image_id] != video_id:
                raise ValueError("physical GT image belongs to multiple videos")
            if image_id in image_frame and image_frame[image_id] != frame_index:
                raise ValueError("physical GT image has multiple frame indices")
            image_video[image_id] = video_id
            image_frame[image_id] = frame_index
            by_video_frame[video_id][frame_index].append((track_key, [x, y, x + width, y + height]))
            count += 1
        by_video_track_counts[video_id][track_key] = count

    if len(by_video_track_counts) != 407:
        raise ValueError(f"physical GT index covers {len(by_video_track_counts)} videos, expected 407")
    return {video: dict(frames) for video, frames in by_video_frame.items()}, dict(by_video_track_counts)


def _stream_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("*/physical_tracks.jsonl"))
    if len(paths) != 13:
        raise ValueError(f"expected 13 completed physical substreams, found {len(paths)}")
    if paths[0].parent.name != "shard_0_tccoff2":
        raise ValueError("unexpected shard-0 namespace")
    return paths


def _new_video_state() -> dict[str, Any]:
    return {
        "tracks": {},
        "by_frame": defaultdict(list),
        "rows": 0,
        "observations": 0,
        "annotated_observations": 0,
    }


def _finalize_video(
    video_id: int,
    state: dict[str, Any],
    gt_by_frame: dict[int, dict[int, list[tuple[str, list[float]]]]],
    gt_track_counts: dict[int, dict[str, int]],
) -> dict[str, int | float]:
    gt_counts = gt_track_counts.get(video_id, {})
    video_gt_by_frame = gt_by_frame.get(video_id, {})
    gt_observations = sum(gt_counts.values())
    pred_tracks = state["tracks"]
    pred_counts = {key: int(value["total"]) for key, value in pred_tracks.items()}
    pred_annotated_counts = Counter()
    matched_observations = 0
    pair_votes: Counter[tuple[str, str]] = Counter()
    pair_iou_sum: Counter[tuple[str, str]] = Counter()
    pair_shared: Counter[tuple[str, str]] = Counter()

    for frame_index, predictions in state["by_frame"].items():
        gt_rows = video_gt_by_frame.get(frame_index, [])
        if not gt_rows or not predictions:
            continue
        gt_keys = [key for key, _box in gt_rows]
        gt_boxes = [box for _key, box in gt_rows]
        pred_keys = [key for key, _box in predictions]
        pred_boxes = [box for _key, box in predictions]
        matrix = _box_iou_matrix(gt_boxes, pred_boxes)
        for gt_index, gt_key in enumerate(gt_keys):
            for pred_index, pred_key in enumerate(pred_keys):
                pair = (pred_key, gt_key)
                pair_shared[pair] += 1
                pair_iou_sum[pair] += float(matrix[gt_index, pred_index])
        bonus = max(matrix.shape) + 1
        weights = np.where(matrix >= IOU_THRESHOLD, bonus + matrix, 0.0)
        gt_indices, pred_indices = linear_sum_assignment(weights, maximize=True)
        for gt_index, pred_index in zip(gt_indices, pred_indices):
            if weights[gt_index, pred_index] <= 0:
                continue
            pred_key = pred_keys[int(pred_index)]
            gt_key = gt_keys[int(gt_index)]
            pair_votes[(pred_key, gt_key)] += 1
            pred_annotated_counts[pred_key] += 1
            matched_observations += 1

    temporal_pairs: list[tuple[float, str, str]] = []
    for pair, shared in pair_shared.items():
        pred_key, gt_key = pair
        denominator = pred_counts[pred_key] + gt_counts[gt_key] - int(shared)
        if denominator <= 0:
            continue
        score = float(pair_iou_sum[pair] / denominator)
        if score >= IOU_THRESHOLD:
            temporal_pairs.append((score, pred_key, gt_key))

    # A deterministic non-overlap diagnostic.  Official MOT/OCD metrics are
    # intentionally not computed here; this only summarizes geometry support.
    temporal_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_pred: set[str] = set()
    used_gt: set[str] = set()
    reliable_pairs: list[tuple[str, str]] = []
    for score, pred_key, gt_key in temporal_pairs:
        if pred_key in used_pred or gt_key in used_gt:
            continue
        used_pred.add(pred_key)
        used_gt.add(gt_key)
        reliable_pairs.append((pred_key, gt_key))

    pred_to_gt = defaultdict(set)
    gt_to_pred = defaultdict(set)
    for pred_key, gt_key in pair_votes:
        pred_to_gt[pred_key].add(gt_key)
        gt_to_pred[gt_key].add(pred_key)
    return {
        "videos": 1,
        "predicted_tracks": len(pred_counts),
        "predicted_observations": int(state["observations"]),
        "predicted_observations_on_gt_images": int(state["annotated_observations"]),
        "gt_tracks": len(gt_counts),
        "gt_observations": gt_observations,
        "per_image_matched_observations": matched_observations,
        "predicted_tracks_with_per_image_match": len(pred_to_gt),
        "gt_tracks_with_per_image_match": len(gt_to_pred),
        "predicted_tracks_with_multiple_gt_votes": sum(len(values) > 1 for values in pred_to_gt.values()),
        "gt_tracks_with_multiple_pred_votes": sum(len(values) > 1 for values in gt_to_pred.values()),
        "reliable_temporal_pairs": len(reliable_pairs),
        "reliable_predicted_tracks": len(used_pred),
        "reliable_gt_tracks": len(used_gt),
        "temporal_candidate_pairs": len(temporal_pairs),
        "predicted_tracks_with_any_annotated_observation": len({key for rows in state["by_frame"].values() for key, _box in rows}),
    }


def _aggregate_video_results(rows: Iterable[dict[str, int | float]]) -> dict[str, int | float]:
    totals: Counter[str] = Counter()
    for row in rows:
        for key, value in row.items():
            if key == "videos":
                totals[key] += int(value)
            elif isinstance(value, int):
                totals[key] += value
    matched = totals["per_image_matched_observations"]
    pred_on_gt = totals["predicted_observations_on_gt_images"]
    gt_obs = totals["gt_observations"]
    gt_tracks = totals["gt_tracks"]
    pred_tracks = totals["predicted_tracks"]
    totals["observation_precision_on_annotated_images"] = matched / max(pred_on_gt, 1)
    totals["observation_recall_against_annotated_gt"] = matched / max(gt_obs, 1)
    totals["reliable_predicted_track_fraction_all"] = totals["reliable_predicted_tracks"] / max(pred_tracks, 1)
    totals["reliable_gt_track_recall"] = totals["reliable_gt_tracks"] / max(gt_tracks, 1)
    return dict(totals)


def audit(gt_path: Path, stream_root: Path, reconciliation_path: Path, output_path: Path, max_videos: int = 0) -> dict[str, Any]:
    gt_by_frame, gt_track_counts = _load_geometry_gt(gt_path)
    paths = _stream_paths(stream_root)
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    if reconciliation.get("status") != "PASS_ALL_407_FIXED_ANONYMOUS_TRAIN_DENSE_SUBSHARDS":
        raise ValueError("full anonymous Train cache reconciliation is not PASS")
    expected_totals = reconciliation.get("totals", {})

    video_results: list[dict[str, int | float]] = []
    seen_videos: set[int] = set()
    seen_tracks: set[tuple[int, str]] = set()
    current_video: int | None = None
    state: dict[str, Any] | None = None
    stream_records: list[dict[str, Any]] = []
    total_rows = total_observations = 0
    last_video_global: int | None = None
    stop_all = False

    def finish_current() -> None:
        nonlocal state, current_video
        if current_video is None or state is None:
            return
        video_results.append(_finalize_video(current_video, state, gt_by_frame, gt_track_counts))
        seen_videos.add(current_video)
        state = None
        current_video = None

    for path in paths:
        digest = hashlib.sha256()
        file_rows = file_observations = 0
        file_first_video = file_last_video = None
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict) or set(row) != PHYSICAL_FIELDS:
                    raise ValueError(f"physical stream schema mismatch at {path}:{line_number}")
                if FORBIDDEN_FIELDS & set(row):
                    raise ValueError(f"semantic/evaluator field found at {path}:{line_number}")
                if row.get("source_split") != "train_predicted":
                    raise ValueError(f"unexpected source split at {path}:{line_number}")
                video_id = int(row["video_id"])
                track_id = str(row["physical_track_id"])
                if video_id not in gt_track_counts:
                    raise ValueError(f"stream video {video_id} is outside the approved geometry index")
                if current_video is None:
                    current_video = video_id
                    state = _new_video_state()
                elif video_id != current_video:
                    if video_id < current_video:
                        raise ValueError("stream video order regressed")
                    if max_videos and len(seen_videos) >= max_videos:
                        finish_current()
                        stop_all = True
                        break
                    finish_current()
                    if max_videos and len(seen_videos) >= max_videos:
                        stop_all = True
                        break
                    if video_id in seen_videos:
                        raise ValueError("video appears in multiple stream positions")
                    current_video = video_id
                    state = _new_video_state()
                if last_video_global is not None and video_id < last_video_global:
                    raise ValueError("global stream video order regressed")
                last_video_global = video_id
                assert state is not None
                track_key = (video_id, track_id)
                if track_key in seen_tracks or track_id in state["tracks"]:
                    raise ValueError(f"duplicate predicted physical track {track_key}")
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
                ):
                    raise ValueError(f"invalid stream lineage at {path}:{line_number}")
                if len(set(frame_ids)) != len(frame_ids):
                    raise ValueError(f"duplicate predicted frame in {track_key}")
                track_record = {"total": len(frame_ids)}
                state["tracks"][track_id] = track_record
                for frame_id, image_path, box in zip(frame_ids, image_paths, boxes):
                    frame_id = int(frame_id)
                    if not _finite_box(box):
                        raise ValueError(f"invalid predicted box at {path}:{line_number}")
                    stem = Path(str(image_path)).stem
                    if not stem.startswith("frame") or not stem[5:].isdigit():
                        raise ValueError(f"predicted image path has no frame number at {path}:{line_number}")
                    frame_index = int(stem[5:]) - 1
                    if frame_index < 0:
                        raise ValueError(f"predicted frame number is not one-based at {path}:{line_number}")
                    if frame_index in gt_by_frame.get(video_id, {}):
                        state["by_frame"][frame_index].append((track_id, [float(value) for value in box]))
                        state["annotated_observations"] += 1
                state["rows"] += 1
                state["observations"] += len(frame_ids)
                file_rows += 1
                file_observations += len(frame_ids)
                total_rows += 1
                total_observations += len(frame_ids)
                file_first_video = video_id if file_first_video is None else file_first_video
                file_last_video = video_id
        stream_records.append({
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
            "rows": file_rows,
            "observations": file_observations,
            "first_video_id": file_first_video,
            "last_video_id": file_last_video,
        })
        if stop_all or (max_videos and len(seen_videos) >= max_videos):
            break
    finish_current()

    full = not max_videos
    result = {
        "schema_version": "trackocd.cvpr.category-free-train-predicted-geometry-audit.v1",
        "status": "COMPLETE_CATEGORY_FREE_TRAIN_PREDICTED_GEOMETRY_MATCH_AUDIT" if full else "COMPLETE_CATEGORY_FREE_TRAIN_PREDICTED_GEOMETRY_MATCH_SMOKE",
        "generated_utc": _now(),
        "scope": {
            "approved_videos": 407,
            "videos_processed": len(video_results),
            "full_fixed_cache_reconciled": full,
            "iou_threshold": IOU_THRESHOLD,
            "matching": "per-image class-free Hungarian at IoU>=0.5; temporal-IoU pair diagnostics and deterministic non-overlap summary",
            "official_trackeval_or_ocd_metrics": False,
        },
        "inputs": {
            "geometry_only_gt_index": str(gt_path.resolve()),
            "geometry_only_gt_index_sha256": _sha256(gt_path),
            "full_cache_reconciliation": str(reconciliation_path.resolve()),
            "full_cache_reconciliation_sha256": _sha256(reconciliation_path),
            "physical_streams": stream_records,
        },
        "counts": {
            "predicted_stream_rows": total_rows,
            "predicted_stream_observations": total_observations,
            "geometry_gt_tracks": sum(len(value) for value in gt_track_counts.values()),
            "geometry_gt_observations": sum(sum(value.values()) for value in gt_track_counts.values()),
        },
        "aggregate_geometry": _aggregate_video_results(video_results),
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
        "interpretation": "Category-free Train geometry audit only. It can qualify coverage/matching evidence for the frontend gate; it is not an OCD/MOT result, semantic label audit, model-selection result, or Goal A/B/C acceptance.",
    }
    if full:
        if total_rows != int(expected_totals.get("tracks", -1)) or total_observations != int(expected_totals.get("observations", -1)):
            raise ValueError("stream totals disagree with the reconciled full cache")
        if len(seen_videos) != 407 or seen_videos != set(gt_track_counts):
            raise ValueError("full stream video coverage does not equal the approved geometry index")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--stream-root", type=Path, default=DEFAULT_STREAM_ROOT)
    parser.add_argument("--reconciliation", type=Path, default=DEFAULT_RECONCILIATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-videos", type=int, default=0, help="bounded smoke limit; zero means all 407 videos")
    args = parser.parse_args()
    if args.max_videos < 0:
        raise SystemExit("--max-videos must be non-negative")
    result = audit(args.geometry_gt.resolve(), args.stream_root.resolve(), args.reconciliation.resolve(), args.output.resolve(), args.max_videos)
    print(json.dumps({
        "status": result["status"],
        "videos_processed": result["scope"]["videos_processed"],
        "counts": result["counts"],
        "aggregate_geometry": result["aggregate_geometry"],
        "boundary": result["boundary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
