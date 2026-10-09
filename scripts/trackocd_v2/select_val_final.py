#!/usr/bin/env python3
"""Freeze the final TrackOCD v2 method from the four-order Val replay."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


TABLE_ROOT = OUTPUT_TARGET / "tables"
FRONTEND_SELECTION = OUTPUT_TARGET / "audit/frontend_selection.json"
REPRESENTATION = OUTPUT_TARGET / "audit/representation_decision.json"
OUTPUT = OUTPUT_TARGET / "audit/val_final_selection.json"
CONTROLLER_OUTPUT = OUTPUT_TARGET / "audit/controller_selection.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise RuntimeError(f"incomplete Val table: {path}: {payload.get('status')}")
    aggregate = payload.get("aggregate", {}).get("16")
    if not aggregate:
        raise RuntimeError(f"p16 aggregate missing from {path}")
    standard = aggregate.get("standard_mean") or {}
    persistent = aggregate.get("persistent_mean") or {}
    required = {"h_score", "commit_ct", "false_assignment_rate"}
    values = {
        "h_score": float(standard["h_score"]),
        "commit_ct": float(persistent["commit_ct"]),
        "false_assignment_rate": float(persistent["false_assignment_rate"]),
    }
    if not all(value == value for value in values.values()):
        raise RuntimeError(f"non-finite Val metrics: {path}")
    return values


def main() -> int:
    out = ensure_output_layout()
    for path, label in ((FRONTEND_SELECTION, "frontend selection"), (REPRESENTATION, "representation decision")):
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    frontend = json.loads(FRONTEND_SELECTION.read_text(encoding="utf-8"))
    representation = json.loads(REPRESENTATION.read_text(encoding="utf-8"))
    if frontend.get("status") != "FINAL_PHYSICAL_FRONTEND_SELECTED":
        raise RuntimeError("Val final selection requires FINAL_PHYSICAL_FRONTEND")
    if representation.get("status") != "FINAL_REPRESENTATION_SELECTED":
        raise RuntimeError("Val final selection requires FINAL_REPRESENTATION_SELECTED")
    baseline_names = ("nearest", "dpmeans", "phe")
    baseline_paths = {name: TABLE_ROOT / f"pred_{name}.json" for name in baseline_names}
    baselines = {name: _metrics(path) for name, path in baseline_paths.items()}
    strongest_name = max(baselines, key=lambda name: (baselines[name]["commit_ct"], baselines[name]["h_score"], -baselines[name]["false_assignment_rate"]))
    strongest = baselines[strongest_name]
    candidate_path = TABLE_ROOT / "pred_safe_controller.json"
    candidate = _metrics(candidate_path) if candidate_path.is_file() else None
    if candidate is not None:
        candidate_safe = (
            candidate["commit_ct"] > strongest["commit_ct"]
            and candidate["false_assignment_rate"] <= strongest["false_assignment_rate"]
            and candidate["h_score"] >= strongest["h_score"] - 0.01
        )
    else:
        candidate_safe = False
    selected_name = "semantic_adapter_safe_controller" if candidate is not None and candidate_safe else strongest_name
    selected_metrics = candidate if selected_name == "semantic_adapter_safe_controller" else strongest
    status = "FINAL_VAL_SELECTION" if candidate is not None and candidate_safe else "FINAL_VAL_SELECTION_NO_IMPROVEMENT"
    controller = {
        "schema_version": "trackocd.v2.controller_selection.v1",
        "status": "FINAL_CONTROLLER_SELECTION",
        "generated_utc": _now(),
        "selected_controller": selected_name,
        "selected_metrics_p16": selected_metrics,
        "incumbent_baseline": strongest_name,
        "candidate": candidate,
        "candidate_safe_against_incumbent": bool(candidate_safe),
        "selection_rule": "candidate Commit-CT strictly above strongest comparable baseline; False Assignment no worse; H-score no worse by more than 0.01; otherwise retain incumbent",
        "test_semantic_accessed": False,
    }
    atomic_json(CONTROLLER_OUTPUT, controller)
    result = {
        "schema_version": "trackocd.v2.val_final_selection.v1",
        "status": status,
        "generated_utc": _now(),
        "final_physical_frontend": frontend.get("selected_frontend"),
        "representation_decision": str(REPRESENTATION.resolve()),
        "baseline_tables": {name: {"path": str(path.resolve()), "sha256": sha256_file(path), "metrics_p16": baselines[name]} for name, path in baseline_paths.items()},
        "strongest_comparable_baseline": {"name": strongest_name, "metrics_p16": strongest},
        "candidate_table": {"path": str(candidate_path.resolve()), "sha256": sha256_file(candidate_path), "metrics_p16": candidate} if candidate_path.is_file() else None,
        "candidate_safe_against_incumbent": bool(candidate_safe),
        "selected_method": selected_name,
        "selected_metrics_p16": selected_metrics,
        "controller_selection": str(CONTROLLER_OUTPUT.resolve()),
        "controller_selection_sha256": sha256_file(CONTROLLER_OUTPUT),
        "selection_rule": controller["selection_rule"],
        "test_semantic_accessed": False,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
