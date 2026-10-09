#!/usr/bin/env python3
"""Promote a completed shared-Val frontend metric run into its stage.

Registration is deliberately separate from metric execution.  A failed or
partial evaluator run can therefore never make a frontend look comparable by
accident.  The category-free route accepts an explicit ``TETA`` not-applicable
status because SimOWT exports only one foreground class; it still requires
the actual TAO-OW ``OWTA``/``AssA`` metrics and both observability diagnostics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.frontend_contract import route_specs  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


STAGES = {
    "simowt": "frontend_simowt.json",
    "ovtr": "frontend_ovtr.json",
    "covtrack_native": "frontend_covtrack_native.json",
    "covtrack_nosem": "frontend_covtrack_nosem.json",
}
FRONTEND_NAMES = {
    "simowt": "SimOWT/Q0",
    "ovtr": "OVTR-native",
    "covtrack_native": "COVTrack-native",
    "covtrack_nosem": "COVTrack-NoSemantic",
}
OVTR_TETA_FAILURE_STATUS = "OVTR_NATIVE_TETA_REFERENCE_EVAL_FAILED"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _teta_status_allowed(frontend: str, status: str) -> bool:
    accepted = {"COMPLETE", "NOT_APPLICABLE_CATEGORY_AGNOSTIC_ROUTE"}
    if frontend == "ovtr":
        accepted.add(OVTR_TETA_FAILURE_STATUS)
    return status in accepted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=tuple(STAGES), required=True)
    parser.add_argument("--metrics-audit", type=Path, default=None)
    args = parser.parse_args()
    out = ensure_output_layout()
    stage_path = out / "audit" / STAGES[args.frontend]
    audit_path = args.metrics_audit or (out / "audit" / f"frontend_{args.frontend}_metrics.json")
    if not stage_path.is_file():
        raise FileNotFoundError(stage_path)
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    stage: dict[str, Any] = json.loads(stage_path.read_text(encoding="utf-8"))
    audit: dict[str, Any] = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "COMPLETE":
        raise RuntimeError(f"frontend metric audit is not complete: {audit.get('status')}")
    tracking = audit.get("tracking") or {}
    observability = audit.get("observability") or {}
    if tracking.get("status") != "COMPLETE":
        raise RuntimeError("category-free TrackEval metrics are not complete")
    if observability.get("status") != "COMPLETE":
        raise RuntimeError("observability metrics are not complete")
    if audit.get("metric_contract", {}).get("gt_join_used_for_model_or_native_stream") is not False:
        raise RuntimeError("metric audit does not prove the GT join was evaluator-only")
    if audit.get("test_semantic_accessed") is not False:
        raise RuntimeError("refusing metric audit with Test semantic access")
    teta_status = str((audit.get("teta_native_reference") or {}).get("status", ""))
    if not _teta_status_allowed(args.frontend, teta_status):
        raise RuntimeError(f"TETA is neither complete nor explicitly not-applicable: {teta_status}")

    frontend_name = FRONTEND_NAMES[args.frontend]
    route = route_specs(ROOT)[frontend_name]
    if args.frontend == "ovtr" and teta_status == OVTR_TETA_FAILURE_STATUS:
        # OVTR TETA is a vocabulary-assisted reference only.  The shared
        # category-free physical metrics above remain the bake-off gate.
        teta_reference_note = "OVTR native TETA reference failed; physical metrics are retained and bake-off is not blocked"
    else:
        teta_reference_note = None

    payload = dict(stage)
    payload.update({
        "schema_version": "trackocd.v2.frontend_stage.v1",
        "status": "BAKEOFF_METRICS_COMPLETE",
        "generated_utc": _now(),
        "native_run_complete": True,
        "same_v2_predicted_stream": False,
        "same_v2_physical_metric_protocol": True,
        "same_v2_evaluator_contract": True,
        "physical_stream_contract_complete": True,
        "requested_metrics_complete": True,
        "historical_reference_only": False,
        "frontend_role": route["frontend_role"],
        "candidate_for_final_frontend": bool(route["candidate_for_final_frontend"]),
        "clean_trackocd_final_frontend": bool(route["clean_trackocd_final_frontend"]),
        "route_contract": route,
        "frontend_metrics_audit": str(audit_path.resolve()),
        "frontend_metrics_audit_sha256": sha256_file(audit_path),
        "frontend_metric_summary": {
            "category_free_tracking": tracking.get("values"),
            "category_free_tracking_percent": tracking.get("values_percent"),
            "teta_native_reference": audit.get("teta_native_reference"),
            "novel_track_observability": observability.get("novel_track_observability"),
            "persistent_observability": observability.get("persistent_observability"),
            "persistent_target_observability": observability.get("persistent_target_observability"),
        },
        "native_metrics_pending": [
            "Standard OCD Old/New/H on the frozen frontend representation",
            "Persistent Commit-CT/False Assignment on the frozen frontend representation",
        ],
        "test_semantic_accessed": False,
        "reason": "The independent native physical stream has completed the shared category-free Val metrics and observability gate; semantic OCD comparison remains downstream of frontend selection.",
    })
    if teta_reference_note:
        payload["reason"] = teta_reference_note
        payload["teta_native_reference_status"] = OVTR_TETA_FAILURE_STATUS
    atomic_json(stage_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
