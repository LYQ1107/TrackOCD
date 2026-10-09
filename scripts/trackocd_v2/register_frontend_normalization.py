#!/usr/bin/env python3
"""Promote a validated normalized stream into a frontend stage artifact.

This records reuse of an already completed native run.  It does not claim
that the shared category-free physical metrics, native TETA reference, or OCD
metrics have been computed; those remain a separate gate in the stage artifact.
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


ASSET_NAMES = {
    "simowt": "SimOWT/Q0",
    "ovtr": "OVTR-native",
    "covtrack_native": "COVTrack-native",
    "covtrack_nosem": "COVTrack-NoSemantic",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=tuple(ASSET_NAMES), required=True)
    parser.add_argument("--normalization-audit", type=Path, default=None)
    args = parser.parse_args()
    out = ensure_output_layout()
    name = ASSET_NAMES[args.frontend]
    route = route_specs(ROOT)[name]
    audit_path = args.normalization_audit or (out / "audit" / f"frontend_{route['slug']}_normalization.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") not in {"COMPLETE", "COMPLETE_WITH_INVALID_BOXES"}:
        raise RuntimeError(f"normalization audit is not complete: {audit_path}")
    contracts = audit.get("contracts", {})
    if contracts.get("physical_stream_contains_category_or_text") is not False:
        raise RuntimeError("refusing to register a physical stream with category/text fields")
    if contracts.get("gt_join_used_for_normalization") is not False:
        raise RuntimeError("refusing a stream normalized with a GT join")
    stage_path = out / "audit" / f"frontend_{args.frontend}.json"
    previous: dict[str, Any] = json.loads(stage_path.read_text(encoding="utf-8")) if stage_path.exists() else {}
    payload = dict(previous)
    payload.update({
        "schema_version": "trackocd.v2.frontend_stage.v1",
        "status": "NATIVE_STREAM_NORMALIZED",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frontend": name,
        "native_run_complete": True,
        "native_run_reused": True,
        "same_v2_predicted_stream": False,
        "same_v2_physical_metric_protocol": False,
        "same_v2_evaluator_contract": True,
        "physical_stream_contract_complete": True,
        "requested_metrics_complete": False,
        "historical_reference_only": False,
        "route_contract": route,
        "normalization_audit": str(audit_path.resolve()),
        "normalization_audit_sha256": sha256_file(audit_path),
        "normalized_outputs": audit.get("outputs"),
        "normalized_counts": audit.get("counts"),
        "native_metrics_pending": [
            "category-free OWTA/AssA/DetRe/LocA",
            "native category-aware TETA reference when applicable",
            "Novel Track Observability",
            "Persistent Target Observability",
            "Persistent Observability",
            "Commit-CT",
            "False Assignment",
        ],
        "test_semantic_accessed": False,
        "reason": "Existing native frontend output was normalized into the category-free v2 physical stream; shared metrics remain pending.",
    })
    atomic_json(stage_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
