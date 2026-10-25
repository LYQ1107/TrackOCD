#!/usr/bin/env python3
"""Audit whether the existing anonymous Train stream is ready for Pred training.

Only compact receipts are opened.  This audit does not read prediction rows,
features, labels, weights, Val/Test assets, or start an optimizer.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FULL_GEOMETRY = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_train_geometry_audit_v1/result.json"
CONTINUITY = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_train_geometry_continuity_reaudit_20261026.json"
POLICY_GEOMETRY = ROOT / "outputs/trackocd_cvpr/cvpr2/policy_predicted_geometry_audit_v1/result.json"
POLICY_INPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1/result.json"
FEATURE_PILOT_72 = ROOT / "outputs/trackocd_cvpr/cvpr2/predicted_train_policy_feature_pilot_v1/result.json"
FEATURE_PILOT_275 = ROOT / "outputs/trackocd_cvpr/cvpr2/predicted_train_policy_feature_pilot_video0275_v1/result.json"
OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr5/predicted_training_readiness_audit_20261026.json"


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "status": payload.get("status")}


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("preserve existing readiness receipt; refusing overwrite")
    full = read(FULL_GEOMETRY)
    continuity = read(CONTINUITY)
    policy = read(POLICY_GEOMETRY)
    policy_input = read(POLICY_INPUT)
    pilot_72 = read(FEATURE_PILOT_72)
    pilot_275 = read(FEATURE_PILOT_275)

    full_geom = full["aggregate_geometry"]
    policy_geom = policy["aggregate_geometry"]
    continuity_values = continuity["continuity"]
    boundaries = [full["boundary"], policy["boundary"], continuity["boundary"]]
    if any(item.get("training_started") is not False for item in boundaries):
        raise ValueError("an input receipt does not prove training was not started")
    if any(item.get("test_accessed") is not False for item in boundaries):
        raise ValueError("an input receipt does not prove TAO Test was not accessed")
    if any(item.get("gt_category_fields_read") is not False for item in boundaries if "gt_category_fields_read" in item):
        raise ValueError("an input receipt does not prove category-free geometry")

    result = {
        "schema_version": "trackocd.cvpr.predicted-training-readiness-audit.v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "BLOCKED_PREDICTED_TRAINING_INPUT_NOT_QUALIFIED",
        "full_train_geometry": {
            "videos": 407,
            "predicted_tracks": full_geom.get("predicted_tracks"),
            "predicted_observations": full_geom.get("predicted_observations"),
            "observation_precision": full_geom.get("observation_precision_on_annotated_images"),
            "observation_recall": full_geom.get("observation_recall_against_annotated_gt"),
            "reliable_predicted_tracks": full_geom.get("reliable_predicted_tracks"),
            "reliable_gt_tracks": full_geom.get("reliable_gt_tracks"),
            "reliable_temporal_pairs": full_geom.get("reliable_temporal_pairs"),
        },
        "full_train_continuity": {
            "gt_adjacent_annotation_pairs": continuity_values.get("gt_adjacent_annotation_pairs"),
            "pairs_with_both_stream_matches": continuity_values.get("pairs_with_both_stream_matches"),
            "same_pred_id_rate_given_both_matches": continuity_values.get("same_pred_id_rate_given_both_matches"),
            "pred_id_switch_rate_given_both_matches": continuity_values.get("pred_id_switch_rate_given_both_matches"),
            "source_frame_gap_one_pairs": continuity_values.get("source_frame_gap_one_pairs"),
            "dominant_source_gaps": continuity_values.get("dominant_source_gaps"),
        },
        "policy_train_geometry": {
            "videos": policy.get("scope", {}).get("videos_processed"),
            "predicted_tracks": policy_geom.get("predicted_tracks"),
            "predicted_observations": policy_geom.get("predicted_observations"),
            "observation_precision": policy_geom.get("observation_precision_on_annotated_images"),
            "observation_recall": policy_geom.get("observation_recall_against_annotated_gt"),
            "reliable_predicted_tracks": policy_geom.get("reliable_predicted_tracks"),
            "reliable_temporal_pairs": policy_geom.get("reliable_temporal_pairs"),
        },
        "interface_and_optimizer_state": {
            "policy_input_status": policy_input.get("status"),
            "policy_input_optimizer_steps": policy_input.get("optimizer_steps"),
            "feature_pilot_video_count": 2,
            "feature_pilot_statuses": [pilot_72.get("status"), pilot_275.get("status")],
            "feature_pilots_are_training": False,
        },
        "source_receipts": [
            receipt(FULL_GEOMETRY, full),
            receipt(CONTINUITY, continuity),
            receipt(POLICY_GEOMETRY, policy),
            receipt(POLICY_INPUT, policy_input),
            receipt(FEATURE_PILOT_72, pilot_72),
            receipt(FEATURE_PILOT_275, pilot_275),
        ],
        "boundary": {
            "category_free_receipts_only": True,
            "prediction_to_gt_mapping_persisted": False,
            "feature_arrays_read": False,
            "training_started": False,
            "optimizer_steps": 0,
            "tao_test_read": False,
            "external_processes_touched": False,
        },
        "interpretation": (
            "The 407-video anonymous stream is complete as provenance but fails "
            "the current predicted-training readiness gate: reliable tracks and "
            "reliable temporal pairs are zero, and the source cadence is not dense "
            "frame-online. The two feature pilots validate only tensor/interface "
            "plumbing. A lawful dense frontend and a non-leaking Train-development "
            "join are required before any Pred optimizer run."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "full_train_geometry": result["full_train_geometry"], "full_train_continuity": result["full_train_continuity"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
