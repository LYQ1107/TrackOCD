#!/usr/bin/env python3
"""Build a Val-only report from completed predicted-track baselines.

This report is deliberately separate from the sealed Test reporting path. It
does not select a method, read Test artifacts, or claim Goal A/B/C acceptance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_v2_runtime"
DEFAULT_REPORT = ROOT / "docs/trackocd_cvpr/VAL_PRED_BASELINE_REPORT_20261022.md"
PREFIXES = (1, 2, 4, 8, 16)
METHODS = ("nearest", "dpmeans")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _check_payload(method: str, payload: dict[str, Any]) -> None:
    if payload.get("status") != "COMPLETE":
        raise RuntimeError(f"{method} baseline is not COMPLETE")
    if payload.get("test_semantic_accessed") is not False:
        raise RuntimeError(f"{method} baseline has Test semantic access")
    stream = payload.get("public_stream") or {}
    if stream.get("source_kind") != "formal_sharded":
        raise RuntimeError(f"{method} baseline is not from the formal sharded cache")
    evaluator = payload.get("evaluator_subset") or {}
    for key in ("fixed_gt_denominator", "public_decisions_filtered_only_after_causal_pass"):
        if evaluator.get(key) is not True:
            raise RuntimeError(f"{method} baseline misses evaluator contract: {key}")
    if evaluator.get("gt_matching_used_as_model_input") is not False:
        raise RuntimeError(f"{method} baseline has GT matching model input")
    contract = payload.get("model_input_contract") or {}
    for key in ("gt_category_id", "gt_split", "temporal_iou_match", "text_or_category_logits"):
        if contract.get(key) is not False:
            raise RuntimeError(f"{method} baseline model contract is unsafe: {key}")


def build(output_root: Path, report_path: Path) -> dict[str, Any]:
    payloads = {method: _load(output_root / "tables" / f"pred_{method}.json") for method in METHODS}
    for method, payload in payloads.items():
        _check_payload(method, payload)
    first = payloads[METHODS[0]]
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        payload = payloads[method]
        aggregate = payload.get("aggregate") or {}
        precise = payload.get("prefixes_result") or {}
        for prefix in PREFIXES:
            key = str(prefix)
            rows.append({
                "method": method,
                "prefix": prefix,
                "standard_mean": aggregate.get(key, {}).get("standard_mean", {}),
                "persistent_mean": aggregate.get(key, {}).get("persistent_mean", {}),
                "standard": (precise.get(key) or {}).get("standard", {}),
                "persistent": (precise.get(key) or {}).get("persistent", {}),
            })
    summary = {
        "schema_version": "trackocd.v2.val_pred_baseline_report.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split": "val",
        "frontend": first.get("public_stream", {}).get("frontend"),
        "source_kind": "formal_sharded",
        "methods": list(METHODS),
        "prefixes": list(PREFIXES),
        "rows": rows,
        "method_selection_performed": False,
        "test_semantic_accessed": False,
        "test_content_accessed": False,
        "goal_abc_acceptance": False,
        "interpretation": "Val-only baseline evidence; no Test or final method-selection claim.",
        "inputs": {
            method: {
                "table": str((output_root / "tables" / f"pred_{method}.json").resolve()),
                "table_sha256": __import__("hashlib").sha256(
                    (output_root / "tables" / f"pred_{method}.json").read_bytes()
                ).hexdigest(),
            }
            for method in METHODS
        },
    }
    output_path = output_root / "tables/val_pred_baseline_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# TrackOCD v2 Val predicted-track baseline report",
        "",
        "Status: `COMPLETE` for the Val-only nearest/DP-means baseline comparison.",
        "",
        "This is evaluator evidence after causal decisions were sealed. It is not a Test result, does not select a final method, and does not establish Goal A/B/C acceptance.",
        "",
        f"Frontend: `{summary['frontend']}`; source: `formal_sharded`; Test semantic access: `false`.",
        "",
        "| Method | Prefix | Old ACC | New ACC | H-score | Commit-CT | False assignment |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        standard = row["standard_mean"]
        persistent = row["persistent_mean"]
        def fmt(value: Any) -> str:
            try:
                return f"{float(value):.6f}"
            except (TypeError, ValueError):
                return "NA"
        lines.append(
            f"| {row['method']} | p{row['prefix']} | {fmt(standard.get('old_acc'))} | "
            f"{fmt(standard.get('new_acc'))} | {fmt(standard.get('h_score'))} | "
            f"{fmt(persistent.get('commit_ct'))} | {fmt(persistent.get('false_assignment_rate'))} |"
        )
    lines.extend([
        "",
        "No method selection or Test access was performed by this report.",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary["report_path"] = str(report_path.resolve())
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build(args.output_root.resolve(), args.report.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
