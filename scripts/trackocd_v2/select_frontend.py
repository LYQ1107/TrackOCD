#!/usr/bin/env python3
"""Apply the v2 frontend-selection gate to the audited assets."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.frontend_contract import (
    CLEAN_PHYSICAL_CANDIDATE_ROLE,
    FRONTENDS,
    OVTR_REFERENCE_ROLE,
    evaluator_contract,
    route_specs,
)  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


AUDIT = OUTPUT_TARGET / "audit/frontend_asset_audit.json"
OUTPUT = OUTPUT_TARGET / "audit/frontend_selection.json"


def _route_for(name: str, stage: dict) -> dict:
    spec = route_specs(ROOT).get(name, {})
    route = stage.get("route_contract")
    if isinstance(route, dict):
        merged = dict(spec)
        merged.update(route)
        return merged
    return spec


def _is_final_frontend_candidate(name: str, stage: dict) -> bool:
    """Keep vocabulary-assisted references out of the clean final choice."""

    route = _route_for(name, stage)
    if route.get("frontend_role") == OVTR_REFERENCE_ROLE:
        return False
    if stage.get("candidate_for_final_frontend") is not None:
        return stage.get("candidate_for_final_frontend") is True
    if route.get("candidate_for_final_frontend") is not None:
        return route.get("candidate_for_final_frontend") is True
    return route.get("frontend_role") == CLEAN_PHYSICAL_CANDIDATE_ROLE


def _value(cell: object) -> float:
    if isinstance(cell, dict):
        value = cell.get("value")
    else:
        value = cell
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return number if number == number else float("-inf")


def _candidate_evidence(name: str, stage: dict) -> dict:
    metrics_path = stage.get("frontend_metrics_audit")
    if not metrics_path:
        return {"frontend": name, "status": "MISSING_METRICS_AUDIT"}
    path = Path(str(metrics_path))
    if not path.is_file():
        return {"frontend": name, "status": "MISSING_METRICS_AUDIT", "path": str(path)}
    metrics = json.loads(path.read_text(encoding="utf-8"))
    tracking = metrics.get("tracking") or {}
    observability = metrics.get("observability") or {}
    values = tracking.get("values") or {}
    return {
        "frontend": name,
        "frontend_role": stage.get("frontend_role") or _route_for(name, stage).get("frontend_role"),
        "candidate_for_final_frontend": _is_final_frontend_candidate(name, stage),
        "status": metrics.get("status"),
        "metrics_audit": str(path.resolve()),
        "category_free_tracking": {key: values.get(key) for key in ("OWTA", "AssA", "DetRe", "LocA")},
        "novel_track_observability": observability.get("novel_track_observability"),
        "persistent_observability": observability.get("persistent_observability"),
        "persistent_target_observability": observability.get("persistent_target_observability"),
        "teta_native_reference": metrics.get("teta_native_reference"),
        "test_semantic_accessed": metrics.get("test_semantic_accessed"),
    }


def main() -> int:
    out = ensure_output_layout()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    stage_paths = {
        name: out / "audit" / filename
        for name, filename in {
            "SimOWT/Q0": "frontend_simowt.json",
            "OVTR-native": "frontend_ovtr.json",
            "COVTrack-native": "frontend_covtrack_native.json",
            "COVTrack-NoSemantic": "frontend_covtrack_nosem.json",
        }.items()
    }
    stages = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in stage_paths.items() if path.exists()}
    missing = sorted(set(FRONTENDS) - set(stages))
    comparable = [
        name for name in FRONTENDS
        if name in stages
        and stages[name].get("native_run_complete") is True
        and stages[name].get("same_v2_physical_metric_protocol") is True
        and stages[name].get("same_v2_evaluator_contract") is True
        and stages[name].get("physical_stream_contract_complete") is True
        and stages[name].get("requested_metrics_complete") is True
    ]
    evidence = {name: _candidate_evidence(name, stages[name]) for name in comparable}
    eligible = [name for name in comparable if _is_final_frontend_candidate(name, stages[name])]
    # Physical selection is category-free: persistent observability is the
    # primary criterion, followed by novel observability and AssA.  TETA is
    # retained as an independent native reference and never substitutes for
    # the category-free tracking metrics.
    ranking = sorted(
        eligible,
        key=lambda name: (
            _value(evidence[name].get("persistent_observability")),
            _value(evidence[name].get("novel_track_observability")),
            _value((evidence[name].get("category_free_tracking") or {}).get("AssA")),
            _value((evidence[name].get("category_free_tracking") or {}).get("OWTA")),
            -list(FRONTENDS).index(name),
        ),
        reverse=True,
    )
    all_required = len(comparable) == len(FRONTENDS)
    selected = ranking[0] if all_required and ranking else None
    result = {
        "schema_version": "trackocd.v2.frontend_selection.v1",
        "status": "FINAL_PHYSICAL_FRONTEND_SELECTED" if selected else "DEFER_FINAL_PHYSICAL_FRONTEND",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selected_frontend": selected,
        "formal_common_feature_cache_authorized": False,
        "requested_frontends": list(FRONTENDS),
        "stage_assets": {name: str(path.resolve()) for name, path in stage_paths.items()},
        "missing_stage_assets": missing,
        "same_protocol_comparable_frontends": comparable,
        "final_frontend_eligible_comparable_frontends": eligible,
        "reference_only_frontends": [name for name in comparable if name not in eligible],
        "candidate_evidence": evidence,
        "ranking": ranking,
        "selection_rule": {
            "required_comparable_frontends": list(FRONTENDS),
            "eligible_final_frontend_roles": ["CLEAN_TRACKOCD_PHYSICAL_CANDIDATE"],
            "excluded_reference_roles": [OVTR_REFERENCE_ROLE],
            "primary": "Persistent Observability",
            "tie_breaks": ["Novel Track Observability", "category-free AssA", "category-free OWTA", "registered frontend order"],
            "teta_is_reference_only": True,
            "ovtr_is_not_a_clean_final_frontend": True,
        },
        "reason": (
            f"Selected {selected} after all requested native frontend streams completed the shared category-free metrics gate."
            if selected else
            "Final selection waits until every requested frontend has an independent native stream and the shared category-free metrics gate."
        ),
        "evaluator_contract": evaluator_contract(ROOT),
        "historical_metrics_do_not_select_frontend": True,
        "source_audit": str(AUDIT.resolve()),
        "source_audit_sha256": sha256_file(AUDIT),
        "test_semantic_accessed": False,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
