#!/usr/bin/env python3
"""Reconcile historical frontend selection with the current local Val receipt.

This is a receipt-only audit.  It opens only compact JSON metadata and never
opens the prediction stream, annotations, weights, or any TAO Test asset.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_SELECTION = ROOT / "outputs/trackocd_cvpr/cvpr2/nas_recovery_20261022/audit/frontend_selection.json"
HISTORICAL_METRICS = ROOT / "outputs/trackocd_cvpr/cvpr2/nas_recovery_20261022/audit/frontend_covtrack_native_metrics.json"
HISTORICAL_STAGE = ROOT / "outputs/trackocd_cvpr/cvpr2/nas_recovery_20261022/audit/frontend_covtrack_native.json"
CURRENT_RECHECK = ROOT / "outputs/trackocd_cvpr/cvpr5/covtrack_native_val_metrics_recheck_20261026.json"
OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr5/frontend_selection_reconciliation_20261026.json"


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


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("preserve existing reconciliation receipt; refusing overwrite")
    selection = read(HISTORICAL_SELECTION)
    historical_metrics = read(HISTORICAL_METRICS)
    historical_stage = read(HISTORICAL_STAGE)
    current = read(CURRENT_RECHECK)

    selected = selection.get("selected_frontend")
    if selected != "COVTrack-native":
        raise ValueError(f"unexpected historical selected frontend: {selected!r}")
    historical_stream = historical_metrics.get("native_stream", {})
    historical_outputs = historical_stage.get("normalized_outputs", {})
    current_stream = current.get("metric_audit", {}).get("native_stream", {})
    historical_physical = historical_stream.get("physical_stream_sha256")
    historical_rows = historical_stream.get("sha256")
    if historical_outputs.get("physical_stream_sha256") != historical_physical:
        raise ValueError("historical metrics/stage physical stream hashes disagree")
    if historical_outputs.get("native_evaluator_rows_sha256") != historical_rows:
        raise ValueError("historical metrics/stage evaluator-row hashes disagree")

    current_physical = current_stream.get("physical_stream_sha256")
    current_rows = current_stream.get("sha256")
    physical_match = historical_physical == current_physical
    rows_match = historical_rows == current_rows
    historical_category_free = historical_stage.get("frontend_metric_summary", {}).get(
        "category_free_tracking", {}
    )
    historical_observability = historical_stage.get("frontend_metric_summary", {})
    current_metric_audit = current.get("metric_audit", {})
    result = {
        "schema_version": "trackocd.cvpr.frontend-selection-reconciliation.v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": (
            "HISTORICAL_SELECTION_NOT_REPROMOTED_CURRENT_STREAM_IDENTITY_MISMATCH"
            if not (physical_match and rows_match)
            else "HISTORICAL_SELECTION_STREAM_IDENTITY_MATCH_RECHECK_ONLY"
        ),
        "historical_selection": {
            "path": str(HISTORICAL_SELECTION.resolve()),
            "sha256": sha256(HISTORICAL_SELECTION),
            "status": selection.get("status"),
            "selected_frontend": selected,
            "formal_common_feature_cache_authorized": selection.get(
                "formal_common_feature_cache_authorized"
            ),
        },
        "historical_stream": {
            "metrics_receipt": str(HISTORICAL_METRICS.resolve()),
            "metrics_receipt_sha256": sha256(HISTORICAL_METRICS),
            "stage_receipt": str(HISTORICAL_STAGE.resolve()),
            "stage_receipt_sha256": sha256(HISTORICAL_STAGE),
            "physical_stream_sha256": historical_physical,
            "native_evaluator_rows_sha256": historical_rows,
            "category_free_tracking": historical_category_free,
            "novel_track_observability": historical_observability.get(
                "novel_track_observability"
            ),
            "persistent_observability": historical_observability.get(
                "persistent_observability"
            ),
        },
        "current_recheck": {
            "path": str(CURRENT_RECHECK.resolve()),
            "receipt_sha256": sha256(CURRENT_RECHECK),
            "status": current.get("status"),
            "physical_stream_sha256": current_physical,
            "native_evaluator_rows_sha256": current_rows,
            "conversion_counts": current_stream.get("conversion_counts"),
            "tracking_values_percent": current_metric_audit.get(
                "tracking_values_percent"
            ),
            "novel_track_observability": current_metric_audit.get(
                "novel_track_observability"
            ),
            "persistent_observability": current_metric_audit.get(
                "persistent_observability"
            ),
        },
        "identity_comparison": {
            "physical_stream_sha256_match": physical_match,
            "native_evaluator_rows_sha256_match": rows_match,
            "same_input_stream_proven": physical_match and rows_match,
        },
        "boundary": {
            "receipt_only": True,
            "large_prediction_stream_opened": False,
            "annotations_opened": False,
            "data_or_weights_read": False,
            "tao_test_read": False,
            "model_selection_performed": False,
            "frontend_repromoted": False,
            "training_started": False,
            "external_processes_touched": False,
        },
        "interpretation": (
            "The historical COVTrack-native selection and the current local Val "
            "recheck refer to different physical/evaluator streams.  The current "
            "category-free metrics remain valid for their own sealed recheck, but "
            "cannot be silently substituted for the historical bakeoff or used to "
            "authorize formal predicted-track training."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "identity_comparison": result["identity_comparison"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
