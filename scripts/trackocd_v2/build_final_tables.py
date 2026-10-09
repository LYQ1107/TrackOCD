#!/usr/bin/env python3
"""Assemble the sealed TrackOCD v2 Val/Test reporting tables.

This is a reporting-only command.  Frontend and method choices must already
be present in the immutable freeze; this script never searches thresholds or
selects a better result from Test.
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

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, atomic_write_text, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.protocol import assert_test_semantic_access_allowed  # noqa: E402


FINAL_FREEZE = OUTPUT_TARGET / "audit/FINAL_FREEZE.json"
FRONTEND_SELECTION = OUTPUT_TARGET / "audit/frontend_selection.json"
REPRESENTATION = OUTPUT_TARGET / "audit/representation_decision.json"
VAL_SELECTION = OUTPUT_TARGET / "audit/val_final_selection.json"
GT_TABLE = OUTPUT_TARGET / "tables/gt_benchmark_table.json"
TEST_OCD = OUTPUT_TARGET / "tables/test_ocd.json"
TEST_PERSISTENT = OUTPUT_TARGET / "audit/test_persistent.json"
STATE = OUTPUT_TARGET / "audit/autonomous_state.json"
FRONTEND_NAMES = {
    "SimOWT/Q0": "simowt",
    "OVTR-native": "ovtr",
    "COVTrack-native": "covtrack_native",
    "COVTrack-NoSemantic": "covtrack_nosem",
}


def _load(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value


def _record(path: Path, label: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "label": label}


def _number(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value")
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _metric_row(value: Any) -> str:
    number = _number(value)
    return "NA" if number is None else f"{number:.6f}"


def _check_prerequisites() -> tuple[dict[str, Any], ...]:
    # This guard intentionally precedes Test table/metrics access.
    assert_test_semantic_access_allowed(FINAL_FREEZE, "build sealed TrackOCD v2 final tables")
    freeze = _load(FINAL_FREEZE, "FINAL_FREEZE")
    selection = _load(FRONTEND_SELECTION, "frontend selection")
    representation = _load(REPRESENTATION, "representation decision")
    val_selection = _load(VAL_SELECTION, "Val final selection")
    gt_table = _load(GT_TABLE, "GT benchmark table")
    test_ocd = _load(TEST_OCD, "Test OCD table")
    test_persistent = _load(TEST_PERSISTENT, "Test persistent audit")
    if freeze.get("status") != "FINAL_FREEZE":
        raise RuntimeError("final tables require FINAL_FREEZE")
    if selection.get("status") != "FINAL_PHYSICAL_FRONTEND_SELECTED":
        raise RuntimeError("final tables require a selected physical frontend")
    if representation.get("status") != "FINAL_REPRESENTATION_SELECTED":
        raise RuntimeError("final tables require a final representation decision")
    if val_selection.get("status") not in {"FINAL_VAL_SELECTION", "FINAL_VAL_SELECTION_NO_IMPROVEMENT"}:
        raise RuntimeError("final tables require a frozen Val method selection")
    if gt_table.get("status") != "COMPLETE":
        raise RuntimeError("GT benchmark table is incomplete")
    if test_ocd.get("status") != "COMPLETE" or test_ocd.get("test_selection_or_tuning") is not False:
        raise RuntimeError("Test OCD table is incomplete or has selection/tuning")
    if test_ocd.get("causal_decisions_sealed_before_test_gt_join") is not True:
        raise RuntimeError("Test OCD table lacks the causal-before-GT boundary")
    if test_persistent.get("status") != "COMPLETE" or test_persistent.get("test_selection_or_tuning") is not False:
        raise RuntimeError("Test persistent audit is incomplete or has selection/tuning")
    frontend = str(freeze.get("frontend") or "")
    slug = FRONTEND_NAMES.get(frontend)
    if slug is None:
        raise RuntimeError(f"unknown frozen frontend: {frontend}")
    test_metrics_path = OUTPUT_TARGET / "audit" / f"frontend_{slug}_test_metrics.json"
    test_metrics = _load(test_metrics_path, "Test frontend metrics")
    if test_metrics.get("status") != "COMPLETE" or test_metrics.get("split") != "test":
        raise RuntimeError("Test frontend tracking metrics are incomplete")
    if test_metrics.get("test_selection_or_tuning") is not False:
        raise RuntimeError("Test frontend metrics do not prove selection/tuning was disabled")
    return freeze, selection, representation, val_selection, gt_table, test_ocd, test_persistent, test_metrics


def _frontend_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    evidence = selection.get("candidate_evidence") or {}
    for name in selection.get("requested_frontends") or []:
        item = evidence.get(name) or {}
        tracking = item.get("category_free_tracking") or {}
        rows.append({
            "frontend": name,
            "status": item.get("status"),
            "owta": _number(tracking.get("OWTA")),
            "assa": _number(tracking.get("AssA")),
            "novel_track_observability": _number(item.get("novel_track_observability")),
            "persistent_observability": _number(item.get("persistent_observability")),
            "persistent_target_observability": _number(item.get("persistent_target_observability")),
            "teta": item.get("teta_native_reference"),
        })
    return rows


def _test_ocd_rows(test_ocd: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in test_ocd.get("method_order") or []:
        payload = (test_ocd.get("methods") or {}).get(method) or {}
        aggregate = payload.get("aggregate") or {}
        p16 = aggregate.get("16") or {}
        standard = p16.get("standard") or {}
        persistent = p16.get("persistent") or {}
        rows.append({
            "method": method,
            "old_acc": _number(standard.get("old_acc")),
            "new_acc": _number(standard.get("new_acc")),
            "h_score": _number(standard.get("h_score")),
            "commit_ct": _number(persistent.get("commit_ct")),
            "false_assignment_rate": _number(persistent.get("false_assignment_rate")),
        })
    if not rows:
        raise RuntimeError("Test OCD table has no method rows")
    return rows


def _markdown(
    *,
    freeze: dict[str, Any],
    frontend_rows: list[dict[str, Any]],
    gt_table: dict[str, Any],
    test_rows: list[dict[str, Any]],
    test_metrics: dict[str, Any],
) -> str:
    lines = [
        "# TrackOCD v2 sealed final tables",
        "",
        f"Frozen physical frontend: **{freeze.get('frontend')}**.",
        "",
        "All frontend selection numbers are Val-only. Test rows are evaluator outputs after the immutable freeze; no Test row was used for selection or tuning.",
        "",
        "## Val physical frontend bake-off",
        "",
        "| Frontend | OWTA | AssA | Novel Observability | Persistent Observability | Persistent Target Observability |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in frontend_rows:
        lines.append(
            f"| {row['frontend']} | {_metric_row(row['owta'])} | {_metric_row(row['assa'])} | "
            f"{_metric_row(row['novel_track_observability'])} | {_metric_row(row['persistent_observability'])} | "
            f"{_metric_row(row['persistent_target_observability'])} |"
        )
    lines.extend([
        "",
        "## Val GT-track OCD baselines",
        "",
        "| Method | Prefix | Old ACC | New ACC | H-score | Commit-CT | False Assignment |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in gt_table.get("rows") or []:
        lines.append(
            f"| {row.get('method')} | p{row.get('prefix')} | {_metric_row(row.get('standard_old_acc'))} | "
            f"{_metric_row(row.get('standard_new_acc'))} | {_metric_row(row.get('standard_h_score'))} | "
            f"{_metric_row(row.get('persistent_commit_ct'))} | {_metric_row(row.get('persistent_false_assignment_rate'))} |"
        )
    lines.extend([
        "",
        "## Frozen TAO Test OCD (p16)",
        "",
        "| Method | Old ACC | New ACC | H-score | Commit-CT | False Assignment |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in test_rows:
        lines.append(
            f"| {row['method']} | {_metric_row(row['old_acc'])} | {_metric_row(row['new_acc'])} | "
            f"{_metric_row(row['h_score'])} | {_metric_row(row['commit_ct'])} | {_metric_row(row['false_assignment_rate'])} |"
        )
    tracking = test_metrics.get("tracking") or {}
    values = tracking.get("values") or {}
    observability = test_metrics.get("observability") or {}
    lines.extend([
        "",
        "## Frozen TAO Test physical tracking",
        "",
        "| OWTA | AssA | DetRe | LocA | Novel Observability | Persistent Observability |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {_metric_row(values.get('OWTA'))} | {_metric_row(values.get('AssA'))} | {_metric_row(values.get('DetRe'))} | {_metric_row(values.get('LocA'))} | "
        f"{_metric_row((observability.get('novel_track_observability') or {}).get('value'))} | "
        f"{_metric_row((observability.get('persistent_observability') or {}).get('value'))} |",
        "",
        "Sealed protocol flags: `test_selection_or_tuning=false`; causal decisions precede the Test GT join; Test semantic evaluation is post-freeze only.",
        "",
    ])
    return "\n".join(lines)


def build() -> dict[str, Any]:
    freeze, selection, representation, val_selection, gt_table, test_ocd, test_persistent, test_metrics = _check_prerequisites()
    frontend_rows = _frontend_rows(selection)
    test_rows = _test_ocd_rows(test_ocd)
    test_metrics_path = OUTPUT_TARGET / "audit" / f"frontend_{FRONTEND_NAMES[str(freeze['frontend'])]}_test_metrics.json"
    payload = {
        "schema_version": "trackocd.v2.final_tables.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frozen_frontend": freeze.get("frontend"),
        "frozen_method": val_selection.get("selected_method"),
        "freeze": _record(FINAL_FREEZE, "FINAL_FREEZE"),
        "frontend_selection": {
            **_record(FRONTEND_SELECTION, "frontend selection"),
            "selected_frontend": selection.get("selected_frontend"),
            "rows": frontend_rows,
        },
        "representation_decision": _record(REPRESENTATION, "representation decision"),
        "val": {
            "final_selection": {
                **_record(VAL_SELECTION, "Val final selection"),
                "selected_method": val_selection.get("selected_method"),
                "selected_metrics_p16": val_selection.get("selected_metrics_p16"),
            },
            "gt_benchmark": {**_record(GT_TABLE, "GT benchmark"), "rows": gt_table.get("rows")},
        },
        "test": {
            "ocd": {**_record(TEST_OCD, "Test OCD"), "rows_p16": test_rows},
            "persistent_audit": _record(TEST_PERSISTENT, "Test persistent audit"),
            "tracking": {
                **_record(test_metrics_path, "Test frontend tracking metrics"),
                "tracking": test_metrics.get("tracking"),
                "observability": test_metrics.get("observability"),
                "teta_native_reference": test_metrics.get("teta_native_reference"),
            },
            "test_selection_or_tuning": False,
            "causal_decisions_sealed_before_test_gt_join": test_ocd.get("causal_decisions_sealed_before_test_gt_join"),
        },
        "state": _record(STATE, "autonomous state") if STATE.is_file() else None,
        "sealed": {
            "test_semantic_accessed": True,
            "test_selection_or_tuning": False,
            "public_or_sealed_status": "SEALED_TEST_EVALUATION",
        },
    }
    atomic_json(OUTPUT_TARGET / "tables/final_tables.json", payload)
    markdown = _markdown(
        freeze=freeze,
        frontend_rows=frontend_rows,
        gt_table=gt_table,
        test_rows=test_rows,
        test_metrics=test_metrics,
    )
    atomic_write_text(ROOT / "docs/trackocd_v2/FINAL_TABLES.md", markdown)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ensure_output_layout()
    result = build()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
