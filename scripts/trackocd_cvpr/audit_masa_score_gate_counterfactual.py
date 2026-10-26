#!/usr/bin/env python3
"""Audit a bounded MASA score-gate counterfactual on sealed Train predictions.

The script never reruns MASA.  It filters the already sealed track observations
at four pre-registered score thresholds and keeps all geometry matching in
memory.  The receipt contains aggregate coverage/purity/continuity values only;
it is a diagnostic and never selects a threshold or creates a formal metric.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from scripts.trackocd_cvpr.audit_masa_covtrack_multi_train_geometry_smoke import (  # noqa: E402
    _load_geometry,
    _max_cardinality_pairs,
)

DEFAULT_CONFIG = ROOT / "configs/trackocd_cvpr/masa_score_gate_counterfactual_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/masa_score_gate_counterfactual_v1/result.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "trackocd.cvpr.masa-score-gate-counterfactual.v1":
        raise ValueError("unexpected MASA score-gate configuration")
    if config.get("status") != "FROZEN_OFFLINE_MASA_SCORE_GATE_COUNTERFACTUAL":
        raise ValueError("MASA score-gate configuration is not frozen")
    thresholds = [float(value) for value in config.get("thresholds", [])]
    if thresholds != sorted(set(thresholds)) or any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("thresholds must be sorted unique probabilities")
    boundaries = config.get("boundaries", {})
    required_false = (
        "new_inference", "training_started", "model_selection",
        "prediction_to_gt_track_mapping_persisted", "gt_category_fields_read",
        "gt_role_or_vocabulary_read", "val_labels_read", "test_accessed",
        "formal_ocd_or_mot_metrics",
    )
    if boundaries.get("existing_sealed_prediction_only") is not True or any(
        boundaries.get(key) is not False for key in required_false
    ):
        raise ValueError("score-gate boundary is not closed")
    return config


def _validate_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = {
        name: (ROOT / relative).resolve()
        for name, relative in config["inputs"].items()
        if name in {"base_result", "worker_input", "sealed_prediction", "geometry_gt_index"}
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing sealed input {name}: {path}")
    expected = config["inputs"]
    for name, key in (
        ("base_result", "expected_base_result_sha256"),
        ("worker_input", "expected_worker_input_sha256"),
        ("sealed_prediction", "expected_sealed_prediction_sha256"),
    ):
        actual = _sha256(paths[name])
        if actual != expected[key]:
            raise ValueError(f"sealed input hash mismatch for {name}")
    return paths


def _evaluate_threshold(
    arrays: Any,
    image_rows: list[dict[str, Any]],
    geometry: dict[int, list[tuple[str, list[float]]]],
    threshold: float,
) -> dict[str, int | float]:
    matched_by_image: dict[int, dict[str, tuple[int, int]]] = {}
    predicted_observations = matched_observations = gt_observations = 0
    for index, row in enumerate(image_rows):
        image_id = int(row["image_id"])
        video_id = int(row["video_id"])
        track_begin = int(arrays["track_offsets"][index])
        track_end = int(arrays["track_offsets"][index + 1])
        boxes = np.asarray(arrays["track_boxes"][track_begin:track_end], dtype=np.float32)
        scores = np.asarray(arrays["track_scores"][track_begin:track_end], dtype=np.float32)
        track_ids = np.asarray(arrays["track_ids"][track_begin:track_end], dtype=np.int64)
        keep = scores >= threshold
        boxes = boxes[keep]
        track_ids = track_ids[keep]
        gt_rows = geometry[image_id]
        gt_boxes = np.asarray([item[1] for item in gt_rows], dtype=np.float32).reshape(-1, 4)
        pairs = _max_cardinality_pairs(gt_boxes, boxes)
        gt_observations += len(gt_boxes)
        predicted_observations += len(boxes)
        matched_observations += len(pairs)
        matched_by_image[image_id] = {
            str(gt_rows[gt_index][0]): (video_id, int(track_ids[pred_index]))
            for gt_index, pred_index in pairs
        }

    by_video: dict[int, list[int]] = defaultdict(list)
    for row in image_rows:
        by_video[int(row["video_id"])].append(int(row["image_id"]))
    annotated_adjacent_pairs = pairs_with_both_matches = same_id_pairs = switched_id_pairs = 0
    for image_ids in by_video.values():
        for previous, current in zip(image_ids, image_ids[1:]):
            previous_gt = {item[0] for item in geometry[previous]}
            current_gt = {item[0] for item in geometry[current]}
            shared = previous_gt & current_gt
            annotated_adjacent_pairs += len(shared)
            for track_id in shared:
                previous_match = matched_by_image[previous].get(track_id)
                current_match = matched_by_image[current].get(track_id)
                if previous_match is None or current_match is None:
                    continue
                pairs_with_both_matches += 1
                if previous_match == current_match:
                    same_id_pairs += 1
                else:
                    switched_id_pairs += 1
    return {
        "threshold": threshold,
        "predicted_track_observations": predicted_observations,
        "matched_track_observations": matched_observations,
        "gt_observations": gt_observations,
        "observation_precision": matched_observations / max(predicted_observations, 1),
        "observation_recall": matched_observations / max(gt_observations, 1),
        "annotated_adjacent_gt_pairs": annotated_adjacent_pairs,
        "pairs_with_both_track_matches": pairs_with_both_matches,
        "same_id_pairs": same_id_pairs,
        "switched_id_pairs": switched_id_pairs,
        "conditional_switch_rate": switched_id_pairs / max(pairs_with_both_matches, 1),
    }


def audit(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    paths = _validate_paths(config)
    base_result = json.loads(paths["base_result"].read_text(encoding="utf-8"))
    if base_result.get("status") != "COMPLETE_CATEGORY_FREE_MULTI_FRONTEND_TRAIN_GEOMETRY_SMOKE_NOT_QUALIFICATION":
        raise ValueError("base MASA receipt is not the sealed category-free geometry smoke")
    worker_input = json.loads(paths["worker_input"].read_text(encoding="utf-8"))
    image_rows = worker_input.get("images")
    if not isinstance(image_rows, list) or len(image_rows) != 32:
        raise ValueError("expected the fixed 32-image Train geometry smoke")
    image_ids = [int(row["image_id"]) for row in image_rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("duplicate image ID in sealed input")
    geometry = _load_geometry(paths["geometry_gt_index"], set(image_ids))
    results: list[dict[str, int | float]] = []
    with np.load(paths["sealed_prediction"], allow_pickle=False) as arrays:
        if [int(value) for value in arrays["image_id"]] != image_ids:
            raise ValueError("sealed prediction image order mismatch")
        if len(arrays["track_scores"]) != len(arrays["track_boxes"]) or len(arrays["track_scores"]) != len(arrays["track_ids"]):
            raise ValueError("sealed track array lengths disagree")
        if not np.isfinite(arrays["track_scores"]).all() or np.any((arrays["track_scores"] < 0) | (arrays["track_scores"] > 1)):
            raise ValueError("sealed track scores are not finite probabilities")
        results = [
            _evaluate_threshold(arrays, image_rows, geometry, float(threshold))
            for threshold in config["thresholds"]
        ]
    result = {
        "schema_version": "trackocd.cvpr.masa-score-gate-counterfactual-result.v1",
        "status": "PASS_MASA_SCORE_GATE_COUNTERFACTUAL_DIAGNOSTIC_NOT_SELECTION",
        "generated_utc": _now(),
        "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
        "inputs": {
            "base_result_sha256": _sha256(paths["base_result"]),
            "worker_input_sha256": _sha256(paths["worker_input"]),
            "sealed_prediction_sha256": _sha256(paths["sealed_prediction"]),
            "fixed_videos": len({int(row["video_id"]) for row in image_rows}),
            "fixed_images": len(image_rows),
        },
        "thresholds": results,
        "boundaries": config["boundaries"],
        "interpretation": "The fixed cuts expose a coverage-purity trade-off over sealed MASA tracks. No threshold is selected, no raw prediction or prediction-to-GT mapping is written, and no formal OCD/MOT result is created.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.config.resolve(), args.output.resolve())
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.resolve()),
        "thresholds": result["thresholds"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
