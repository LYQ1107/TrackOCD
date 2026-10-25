#!/usr/bin/env python3
"""Extract a compact, fail-closed receipt from the Val metrics recheck."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "trackocd.cvpr.covtrack-native-val-metrics-recheck.v1"
METRICS = "outputs/trackocd_v2_runtime/audit/frontend_covtrack_native_metrics_20261026_recheck.json"
GT_AUDIT = "outputs/trackocd_v2_runtime/audit/val_trackeval_gt.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_audit(repo: Path) -> dict[str, Any]:
    metrics_path = repo / METRICS
    gt_audit_path = repo / GT_AUDIT
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    gt_audit = json.loads(gt_audit_path.read_text(encoding="utf-8"))
    tracking = metrics.get("tracking") or {}
    observability = metrics.get("observability") or {}
    contract = metrics.get("metric_contract") or {}
    teta = metrics.get("teta_native_reference") or {}
    values = tracking.get("values_percent") or {}
    novel = observability.get("novel_track_observability") or {}
    persistent = observability.get("persistent_observability") or {}
    native_stream = metrics.get("native_stream") or {}
    conversion = native_stream.get("conversion") or {}
    native_stream_summary = {
        "path": native_stream.get("path"),
        "sha256": native_stream.get("sha256"),
        "physical_stream_path": native_stream.get("physical_stream_path"),
        "physical_stream_sha256": native_stream.get("physical_stream_sha256"),
        "conversion_counts": {
            key: conversion.get(key) for key in ("rows", "physical_tracks", "videos")
        },
    }
    teta_failure = teta.get("status") == "NATIVE_TETA_REFERENCE_EVAL_FAILED"
    safe = (
        metrics.get("status") == "COMPLETE"
        and metrics.get("split") == "val"
        and metrics.get("test_semantic_accessed") is False
        and metrics.get("test_selection_or_tuning") is False
        and tracking.get("status") == "COMPLETE"
        and observability.get("status") == "COMPLETE"
        and contract.get("gt_join_used_for_model_or_native_stream") is False
        and contract.get("gt_join_used_for_scoring_only") is True
        and teta_failure
        and teta.get("physical_metrics_preserved") is True
        and gt_audit.get("status") == "COMPLETE"
        and gt_audit.get("split") == "val"
        and gt_audit.get("test_semantic_accessed") is False
    )
    return {
        "schema_version": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS_VAL_CATEGORY_FREE_METRICS_RECHECK_TETA_PENDING_NOT_FRONTEND_ACCEPTANCE"
            if safe
            else "INCOMPLETE_VAL_METRICS_RECHECK"
        ),
        "scope": {
            "val_only": True,
            "large_prediction_stream_reopened": False,
            "data_or_weights_read": False,
            "tao_test_read": False,
            "model_selection_performed": False,
            "training_started": False,
            "external_processes_touched": False,
        },
        "metric_audit": {
            "path": METRICS,
            "bytes": metrics_path.stat().st_size,
            "sha256": sha256_file(metrics_path),
            "run_id": metrics.get("run_id"),
            "native_stream": native_stream_summary,
            "tracking_status": tracking.get("status"),
            "tracking_values_percent": {
                key: values.get(key)
                for key in ("HOTA", "AssA", "DetA", "DetPr", "DetRe", "AssPr", "AssRe", "LocA", "OWTA")
            },
            "novel_track_observability": novel,
            "persistent_observability": persistent,
            "teta_reference_status": teta.get("status"),
            "teta_failure_reason": teta.get("error"),
            "physical_metrics_preserved": teta.get("physical_metrics_preserved"),
        },
        "val_trackeval_gt": {
            "audit_path": GT_AUDIT,
            "bytes": gt_audit_path.stat().st_size,
            "sha256": sha256_file(gt_audit_path),
            "output": gt_audit.get("output"),
            "output_sha256": gt_audit.get("output_sha256"),
            "counts": gt_audit.get("counts"),
        },
        "qualification_boundary": {
            "category_free_metrics_are_real": safe,
            "teta_reference_complete": False,
            "frontend_registered_as_bakeoff_complete": False,
            "frontend_selection_performed": False,
            "formal_predicted_cache_authorized": False,
            "goal_a_b_c_accepted": False,
        },
        "scientific_boundary": "This is a Val category-free MOT/observability recheck on the sealed COVTrack-native stream. TETA is reference-only and unavailable in the current environment; the result is not frontend selection, predicted-track training, formal OCD acceptance, or Goal A/B/C success.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trackocd_cvpr/cvpr5/covtrack_native_val_metrics_recheck_20261026.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    result = build_audit(repo)
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0 if result["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
