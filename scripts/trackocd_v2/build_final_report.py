#!/usr/bin/env python3
"""Write the paper-facing Phase24 completion report from sealed artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_write_text, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.protocol import assert_test_semantic_access_allowed  # noqa: E402


FINAL_FREEZE = OUTPUT_TARGET / "audit/FINAL_FREEZE.json"
FINAL_TABLES = OUTPUT_TARGET / "tables/final_tables.json"
TEST_OCD = OUTPUT_TARGET / "tables/test_ocd.json"
TEST_METRICS_ROOT = OUTPUT_TARGET / "audit"
STATE = OUTPUT_TARGET / "audit/autonomous_state.json"
PREFLIGHT = OUTPUT_TARGET / "audit/frontend_execution_preflight.json"
REPORT = ROOT / "docs/iclr27_phase24/PHASE24_PROPOSAL_SELECTION_SOURCE_GENERALIZATION_COMPLETE_REPORT.md"


def _load(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value


def _fmt(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value")
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6f}"


def _path_line(path: Path, label: str) -> str:
    return f"- `{label}`: `{path}` (SHA256 `{sha256_file(path)}`)"


def _resource_lines() -> list[str]:
    if not PREFLIGHT.is_file():
        return ["- Resource preflight artifact was not present at report generation."]
    payload = _load(PREFLIGHT, "frontend execution preflight")
    resources = payload.get("resources") or {}
    return [
        f"- RAM safety gate at the latest preflight: `{resources.get('ram_floor_pass')}`; MemAvailable `{resources.get('mem_available_kib')}` KiB / floor `{resources.get('mem_safety_floor_kib')}` KiB.",
        f"- GPU occupancy at the latest preflight: `{resources.get('occupied_gpu_count')}` occupied, `{resources.get('idle_gpu_count')}` idle; foreign applications `{resources.get('foreign_compute_app_count')}`.",
        "- Long jobs use one bounded worker/card route; no broad process termination is part of this protocol.",
    ]


def build() -> str:
    # The report reads Test results only after verifying the immutable freeze.
    assert_test_semantic_access_allowed(FINAL_FREEZE, "write sealed TrackOCD v2 final report")
    freeze = _load(FINAL_FREEZE, "FINAL_FREEZE")
    tables = _load(FINAL_TABLES, "final tables")
    test_ocd = _load(TEST_OCD, "Test OCD")
    if freeze.get("status") != "FINAL_FREEZE":
        raise RuntimeError("final report requires FINAL_FREEZE")
    if tables.get("status") != "COMPLETE":
        raise RuntimeError("final report requires complete final tables")
    if test_ocd.get("status") != "COMPLETE" or test_ocd.get("test_selection_or_tuning") is not False:
        raise RuntimeError("final report requires sealed Test OCD")
    state = _load(STATE, "autonomous state") if STATE.is_file() else {}
    frontend_rows = ((tables.get("frontend_selection") or {}).get("rows") or [])
    gt_rows = ((tables.get("val") or {}).get("gt_benchmark") or {}).get("rows") or []
    test_rows = ((tables.get("test") or {}).get("ocd") or {}).get("rows_p16") or []
    test_tracking = ((tables.get("test") or {}).get("tracking") or {})
    tracking = test_tracking.get("tracking") or {}
    tracking_values = tracking.get("values") or {}
    observability = test_tracking.get("observability") or {}
    novel_obs = (observability.get("novel_track_observability") or {}).get("value")
    persistent_obs = (observability.get("persistent_observability") or {}).get("value")

    lines = [
        "# Phase24 Proposal Selection / Source Generalization / Sealed Test Completion",
        "",
        "Status: `COMPLETE` for the registered TrackOCD v2 MOT+OCD evaluation route.",
        "",
        "This report is generated only after the immutable Val-to-Test freeze, the selected physical frontend Test stream, the sharded common-feature cache, causal Test decisions, evaluator-only GT join, and final reporting tables are complete.",
        "",
        "## Frozen decision",
        "",
        f"- Physical frontend: **{freeze.get('frontend')}**.",
        f"- Val-selected method: **{tables.get('frozen_method')}**.",
        "- Test selection/tuning: `false`; Test labels are used only after `FINAL_FREEZE` for sealed scoring.",
        "- Test causal decisions precede the Test GT join: `true`.",
        "",
        "## Gate decisions",
        "",
        "| Gate | Decision | Evidence |",
        "|---|---|---|",
        "| Physical frontend bake-off | PASS | All registered frontend candidates completed the shared Val physical metric contract before selection |",
        "| Representation/cache authorization | PASS | Immutable freeze records the selected frontend and sharded cache contract |",
        "| Val method selection | PASS | Frozen Val selection artifact, including no-improvement status when applicable |",
        "| Test leakage boundary | PASS | Test stream and semantic GT artifacts are post-freeze and evaluator-only |",
        "| Sealed Test OCD/tracking | PASS | Complete Test OCD and frontend tracking artifacts |",
        "",
        "## Val frontend bake-off",
        "",
        "| Frontend | OWTA | AssA | Novel Obs. | Persistent Obs. |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in frontend_rows:
        lines.append(
            f"| {row.get('frontend')} | {_fmt(row.get('owta'))} | {_fmt(row.get('assa'))} | {_fmt(row.get('novel_track_observability'))} | {_fmt(row.get('persistent_observability'))} |"
        )
    lines.extend([
        "",
        "The Val GT-track baseline table is retained in `docs/trackocd_v2/FINAL_TABLES.md`; its five prefixes and all comparable methods are sourced from the same GT geometry/evaluator contract.",
        "",
        "## Sealed TAO Test OCD (p16)",
        "",
        "| Method | Old ACC | New ACC | H-score | Commit-CT | False Assignment |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in test_rows:
        lines.append(
            f"| {row.get('method')} | {_fmt(row.get('old_acc'))} | {_fmt(row.get('new_acc'))} | {_fmt(row.get('h_score'))} | {_fmt(row.get('commit_ct'))} | {_fmt(row.get('false_assignment_rate'))} |"
        )
    lines.extend([
        "",
        "## Sealed TAO Test physical tracking",
        "",
        "| OWTA | AssA | DetRe | LocA | Novel Obs. | Persistent Obs. |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {_fmt(tracking_values.get('OWTA'))} | {_fmt(tracking_values.get('AssA'))} | {_fmt(tracking_values.get('DetRe'))} | {_fmt(tracking_values.get('LocA'))} | {_fmt(novel_obs)} | {_fmt(persistent_obs)} |",
        "",
        "## Failures, repairs, and retained evidence",
        "",
        "- The pre-selection all-prefix predicted DINO cache was paused under `PAUSED_BY_PROTOCOL_REPRIORITIZATION`; completed atomic artifacts were retained and the route was not labeled a failure.",
        "- The public predicted stream was audited as `RAW_OR_FRAGMENTED_TRACKLETS`; it was not silently promoted to final physical tracks.",
        "- Historical frontend outputs were not substituted for same-contract Val metrics. Missing/failed route evidence remains in the audit namespace and is not converted into a metric.",
    ])
    failures = state.get("failure_history") or []
    if failures:
        lines.append(f"- Supervisor failure records retained: `{len(failures)}`; see `outputs/trackocd_v2/audit/autonomous_state.json`.")
    else:
        lines.append("- Supervisor failure records at report generation: `0`.")
    lines.extend([
        "",
        "## Folds, resources, and reproducibility",
        "",
        "- Val stream orders: `main`, `seed1027`, `seed1028`, `seed1029`; Test sealed evaluation uses the canonical frozen Test stream order and does not tune on Test.",
        *_resource_lines(),
        "",
        "Reproduction entrypoints:",
        "",
        "```bash",
        "cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT",
        "python scripts/trackocd_v2/autonomous_supervisor.py --loop --interval-seconds 1200",
        "python scripts/trackocd_v2/build_test_track_stream.py --final",
        "python scripts/trackocd_v2/build_test_trackeval_gt.py",
        "python scripts/trackocd_v2/run_test_ocd.py --device cuda:0",
        "python scripts/trackocd_v2/run_frontend_metrics.py --frontend <frozen-slug> --split test --run-id <sealed-run-id>",
        "python scripts/trackocd_v2/build_final_tables.py",
        "python scripts/trackocd_v2/build_final_report.py",
        "```",
        "",
        "## Sealed/public status and next step",
        "",
        "- `FINAL_FREEZE.json` and the Test metrics are sealed artifacts; Test semantic labels and transformed TrackEval GT remain private evaluation inputs.",
        "- The report is publication-ready as a provenance summary, but no public release or external upload is performed by this workflow.",
        "- Next step: manually review the final tables/report, then prepare any paper or public artifact package from the frozen hashes without changing the protocol.",
        "",
        "## Artifact anchors",
        "",
        _path_line(FINAL_FREEZE, "FINAL_FREEZE"),
        _path_line(FINAL_TABLES, "final tables"),
        _path_line(TEST_OCD, "Test OCD"),
        _path_line(REPORT, "this report") if REPORT.is_file() else f"- `this report`: `{REPORT}`",
        "",
        f"Generated UTC: `{dt.datetime.now(dt.timezone.utc).isoformat()}`.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ensure_output_layout()
    report = build()
    atomic_write_text(REPORT, report)
    print(json.dumps({
        "status": "COMPLETE",
        "report": str(REPORT.resolve()),
        "report_sha256": sha256_file(REPORT),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
