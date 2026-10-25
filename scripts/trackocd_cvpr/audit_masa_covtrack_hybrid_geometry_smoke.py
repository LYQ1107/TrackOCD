#!/usr/bin/env python3
"""Evaluate one frozen MASA/COVTrack union rule on an existing Train smoke.

All prediction payloads are loaded before the category-free geometry index is
opened.  The script writes compact aggregate evidence only; no GT mapping or
filtered prediction stream is persisted.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/trackocd_cvpr/masa_covtrack_hybrid_geometry_smoke_v1.json"
BASE_CONFIG = ROOT / "configs/trackocd_cvpr/masa_covtrack_multi_train_geometry_smoke_v1.json"
BASE_RECEIPT = ROOT / "outputs/trackocd_cvpr/cvpr2/masa_covtrack_multi_train_geometry_smoke_v1/result.json"
MASA_PREDICTION = ROOT / "outputs/trackocd_cvpr/cvpr2/masa_covtrack_multi_train_geometry_smoke_v1/run/sealed_prediction.npz"
COVTRACK_STREAM = ROOT / "outputs/trackocd_v2_runtime/manifests/frontend_streams/covtrack_native/covtrack_train_dense_shards_v1/shard_0_tccoff2/physical_tracks.jsonl"
GEOMETRY = ROOT / "outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v10_physical_GT/private_exact_joint_proof_or_assignment.json"
OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/masa_covtrack_hybrid_geometry_smoke_v1/result.json"

PHYSICAL_FIELDS = {
    "sample_key", "source_sample_id", "video_id", "physical_track_id", "frame_ids",
    "boxes_xyxy", "image_paths", "quality", "stream_order", "source_split",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    return intersection / max(left_area + right_area - intersection, 1e-12)


def _pairs(gt_rows: list[tuple[str, np.ndarray]], predictions: list[tuple[str, np.ndarray]], threshold: float) -> list[tuple[int, int]]:
    if not gt_rows or not predictions:
        return []
    matrix = np.asarray([[_iou(gbox, pbox) for _, pbox in predictions] for _, gbox in gt_rows], dtype=np.float32)
    rows, columns = linear_sum_assignment(np.where(matrix >= threshold, 1.0 - matrix, 1000.0))
    return [(int(row), int(column)) for row, column in zip(rows, columns) if matrix[row, column] >= threshold]


def _fixed_input() -> dict[str, Any]:
    base = json.loads(BASE_RECEIPT.read_text(encoding="utf-8"))
    images = base["selection"]["images"]
    if len(images) != 32 or len({int(row["image_id"]) for row in images}) != 32:
        raise ValueError("base smoke does not contain the fixed 32-image selection")
    return {
        "images": images,
        "videos": [int(value) for value in base["selection"]["videos"]],
        "image_plan_sha256": base["selection"]["image_plan_sha256"],
    }


def _load_masa(image_order: list[int], video_by_image: dict[int, int]) -> dict[int, list[tuple[str, np.ndarray]]]:
    if sha256_file(MASA_PREDICTION) != json.loads(CONFIG.read_text())["sealed_masa_prediction_sha256"]:
        raise ValueError("sealed MASA prediction hash mismatch")
    with np.load(MASA_PREDICTION, allow_pickle=False) as arrays:
        observed = [int(value) for value in arrays["image_id"]]
        if observed != image_order:
            raise ValueError("MASA image order differs from fixed input")
        offsets = arrays["track_offsets"].astype(np.int64)
        boxes = arrays["track_boxes"].astype(np.float32)
        ids = arrays["track_ids"].astype(np.int64)
        result: dict[int, list[tuple[str, np.ndarray]]] = {}
        for index, image_id in enumerate(observed):
            begin, end = int(offsets[index]), int(offsets[index + 1])
            result[image_id] = [
                (f"masa:{video_by_image[image_id]}:{int(ids[position])}", boxes[position].copy())
                for position in range(begin, end)
            ]
        return result


def _load_covtrack(images: list[dict[str, Any]]) -> tuple[dict[int, list[tuple[str, np.ndarray]]], int, int]:
    path_to_image = {str(row["image_path"]): int(row["image_id"]) for row in images}
    result: dict[int, list[tuple[str, np.ndarray]]] = {image_id: [] for image_id in path_to_image.values()}
    rows_scanned = target_hits = 0
    with COVTRACK_STREAM.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            rows_scanned += 1
            if set(row) != PHYSICAL_FIELDS or row.get("source_split") != "train_predicted":
                raise ValueError("COVTrack stream is not the registered anonymous Train contract")
            for image_path, box in zip(row["image_paths"], row["boxes_xyxy"]):
                if image_path not in path_to_image:
                    continue
                values = np.asarray(box, dtype=np.float32)
                if values.shape != (4,) or not np.isfinite(values).all() or np.any(values[2:] <= values[:2]):
                    raise ValueError("invalid COVTrack geometry")
                image_id = path_to_image[image_path]
                result[image_id].append((f"covtrack:{row['physical_track_id']}", values))
                target_hits += 1
    if any(not rows for rows in result.values()):
        raise ValueError("COVTrack stream does not cover every fixed image")
    return result, rows_scanned, target_hits


def _load_geometry(image_ids: set[int]) -> dict[int, list[tuple[str, np.ndarray]]]:
    payload = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    if payload.get("semantic_category_labels_retained") is not False or payload.get("not_model_facing") is not True:
        raise ValueError("geometry input is not the sealed category-free index")
    result: dict[int, list[tuple[str, np.ndarray]]] = {image_id: [] for image_id in image_ids}
    for track in payload.get("tracks", []):
        if set(track) != {"video_id", "opaque_track_id", "observations"}:
            raise ValueError("geometry input contains non-geometry fields")
        for observation in track["observations"]:
            image_id = int(observation[0])
            if image_id not in result:
                continue
            x, y, width, height = map(float, observation[2:6])
            box = np.asarray([x, y, x + width, y + height], dtype=np.float32)
            if not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                raise ValueError("invalid category-free geometry")
            result[image_id].append((str(track["opaque_track_id"]), box))
    return result


def _hybrid(covtrack: list[tuple[str, np.ndarray]], masa: list[tuple[str, np.ndarray]], threshold: float) -> list[tuple[str, np.ndarray]]:
    retained = [(identity, box.copy()) for identity, box in covtrack]
    for identity, box in masa:
        if not retained or max(_iou(box, existing) for _, existing in retained) < threshold:
            retained.append((identity, box.copy()))
    return retained


def _summarize(name: str, streams: dict[int, list[tuple[str, np.ndarray]]], image_order: list[int], by_video: dict[int, list[int]], geometry: dict[int, list[tuple[str, np.ndarray]]], threshold: float) -> dict[str, Any]:
    matched_by_image: dict[int, dict[str, str]] = {}
    counts = {"gt_observations": 0, "predicted_observations": 0, "matched_observations": 0}
    gt_votes: dict[str, set[str]] = defaultdict(set)
    predicted_votes: dict[str, set[str]] = defaultdict(set)
    for image_id in image_order:
        gt_rows = geometry[image_id]
        predictions = streams[image_id]
        matches = _pairs(gt_rows, predictions, threshold)
        counts["gt_observations"] += len(gt_rows)
        counts["predicted_observations"] += len(predictions)
        counts["matched_observations"] += len(matches)
        matched_by_image[image_id] = {gt_rows[g][0]: predictions[p][0] for g, p in matches}
        for g, p in matches:
            gt_votes[gt_rows[g][0]].add(predictions[p][0])
            predicted_votes[predictions[p][0]].add(gt_rows[g][0])
    temporal = {"annotated_adjacent_gt_pairs": 0, "pairs_with_both_track_matches": 0, "same_id_pairs": 0, "switched_id_pairs": 0}
    for video_id in by_video:
        frames = by_video[video_id]
        for previous, current in zip(frames, frames[1:]):
            shared = {track_id for track_id, _ in geometry[previous]} & {track_id for track_id, _ in geometry[current]}
            temporal["annotated_adjacent_gt_pairs"] += len(shared)
            for track_id in shared:
                if track_id not in matched_by_image[previous] or track_id not in matched_by_image[current]:
                    continue
                temporal["pairs_with_both_track_matches"] += 1
                if matched_by_image[previous][track_id] == matched_by_image[current][track_id]:
                    temporal["same_id_pairs"] += 1
                else:
                    temporal["switched_id_pairs"] += 1
    return {
        "name": name,
        "gt_observations": counts["gt_observations"],
        "predicted_observations": counts["predicted_observations"],
        "matched_observations": counts["matched_observations"],
        "observation_precision": counts["matched_observations"] / max(counts["predicted_observations"], 1),
        "observation_recall": counts["matched_observations"] / max(counts["gt_observations"], 1),
        "gt_tracks": len(gt_votes),
        "gt_tracks_with_any_match": sum(bool(values) for values in gt_votes.values()),
        "gt_tracks_with_multiple_pred_votes": sum(len(values) > 1 for values in gt_votes.values()),
        "predicted_tracks": len(predicted_votes),
        "predicted_tracks_with_any_gt_vote": sum(bool(values) for values in predicted_votes.values()),
        "predicted_tracks_with_multiple_gt_votes": sum(len(values) > 1 for values in predicted_votes.values()),
        "temporal": {
            **temporal,
            "conditional_switch_rate": temporal["switched_id_pairs"] / max(temporal["pairs_with_both_track_matches"], 1),
            "conditional_same_id_rate": temporal["same_id_pairs"] / max(temporal["pairs_with_both_track_matches"], 1),
        },
    }


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != "trackocd.cvpr.masa-covtrack-hybrid-geometry-smoke.v1":
        raise ValueError("unexpected hybrid configuration")
    if sha256_file(BASE_CONFIG) != cfg["base_comparison_config_sha256"]:
        raise ValueError("base comparison configuration changed")
    if sha256_file(GEOMETRY) != cfg["inputs"]["geometry_only_gt_index_sha256"]:
        raise ValueError("geometry index changed")
    if OUTPUT.exists():
        raise ValueError("preserve existing hybrid receipt; refusing overwrite")
    fixed = _fixed_input()
    image_order = [int(row["image_id"]) for row in fixed["images"]]
    by_video: dict[int, list[int]] = defaultdict(list)
    video_by_image = {}
    for row in fixed["images"]:
        image_id, video_id = int(row["image_id"]), int(row["video_id"])
        by_video[video_id].append(image_id)
        video_by_image[image_id] = video_id
    # Prediction inputs are fully loaded before the evaluator-only geometry is opened.
    masa = _load_masa(image_order, video_by_image)
    covtrack, rows_scanned, target_hits = _load_covtrack(fixed["images"])
    geometry = _load_geometry(set(image_order))
    threshold = float(cfg["fusion_rule"]["iou_threshold"])
    streams = {
        "covtrack_native": covtrack,
        "masa_native": masa,
        "covtrack_first_union_dedup": {image_id: _hybrid(covtrack[image_id], masa[image_id], threshold) for image_id in image_order},
    }
    result = {
        "schema_version": "trackocd.cvpr.masa-covtrack-hybrid-geometry-smoke-result.v1",
        "status": "COMPLETE_CATEGORY_FREE_HYBRID_GEOMETRY_SMOKE_NOT_QUALIFICATION",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {"path": str(CONFIG.resolve()), "sha256": sha256_file(CONFIG)},
        "inputs": {
            "base_receipt": str(BASE_RECEIPT.resolve()),
            "base_receipt_sha256": sha256_file(BASE_RECEIPT),
            "fixed_images": len(image_order),
            "videos": len(by_video),
            "masa_prediction_sha256": sha256_file(MASA_PREDICTION),
            "covtrack_stream_sha256": sha256_file(COVTRACK_STREAM),
            "covtrack_rows_scanned": rows_scanned,
            "covtrack_target_hits": target_hits,
        },
        "fusion_rule": cfg["fusion_rule"],
        "metrics": [_summarize(name, stream, image_order, by_video, geometry, threshold) for name, stream in streams.items()],
        "boundary": {
            "new_inference": False,
            "training_started": False,
            "gt_category_fields_read": False,
            "gt_role_or_vocabulary_read": False,
            "prediction_to_gt_mapping_persisted": False,
            "val_labels_read": False,
            "test_accessed": False,
            "official_metrics": False,
            "frontend_qualification": False,
        },
        "interpretation": "Fixed same-image Train fusion diagnostic only. The hybrid rule is not selected, does not qualify a frontend, and does not create an OCD/MOT metric or Goal A/B/C result.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
