#!/usr/bin/env python3
"""Register one frontend asset stage without launching a native run."""

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


ASSET_NAMES = {
    "simowt": "SimOWT/Q0",
    "ovtr": "OVTR-native",
    "covtrack_native": "COVTrack-native",
    "covtrack_nosem": "COVTrack-NoSemantic",
}
AUDIT = OUTPUT_TARGET / "audit/frontend_asset_audit.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=tuple(ASSET_NAMES))
    args = parser.parse_args()
    out = ensure_output_layout()
    if not AUDIT.exists():
        raise FileNotFoundError(AUDIT)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    name = ASSET_NAMES[args.frontend]
    asset = audit["historical_assets"][name]
    route = route_specs(ROOT)[name]
    status = str(asset.get("v2_execution", "UNKNOWN"))
    payload: dict[str, Any] = {
        "schema_version": "trackocd.v2.frontend_stage.v1",
        "status": "ASSET_REGISTERED",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frontend": name,
        "stage_input_audit": str(AUDIT.resolve()),
        "stage_input_audit_sha256": sha256_file(AUDIT),
        "asset_status": status,
        "native_run_complete": False,
        "same_v2_predicted_stream": False,
        "same_v2_physical_metric_protocol": False,
        "same_v2_evaluator_contract": False,
        "physical_stream_contract_complete": False,
        "requested_metrics_complete": False,
        "trackocd_backend_category_free": route["trackocd_backend_category_free"],
        "novel_ontology_leakage": route["novel_ontology_leakage"],
        "detector_ontology": route["detector_ontology"],
        "semantic_association_mode": route["semantic_association_mode"],
        "fixed_overrides": route["fixed_overrides"],
        "route_contract": route,
        "historical_reference_only": True,
        "historical_asset": asset,
        "test_semantic_accessed": False,
    }
    if args.frontend == "covtrack_nosem":
        payload.update({
            "status": "NOT_YET_RUN",
            "historical_reference_only": False,
            "reason": "No COVTrack-NoSemantic native output is registered.",
        })
    else:
        payload["reason"] = "This stage records an existing asset only; no new native frontend inference is claimed."
    path = out / "audit" / f"frontend_{args.frontend}.json"
    atomic_json(path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
