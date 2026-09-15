#!/usr/bin/env python3
"""Assemble the comparable GT-track baseline table.

The table is a reporting artifact only.  It reads completed evaluator outputs
after causal decisions have been made and never feeds labels back into a
method.  A legacy current-model contract failure is represented explicitly,
not converted into a metric.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, atomic_write_text, ensure_output_layout, sha256_file  # noqa: E402


PREFIXES = (1, 2, 4, 8, 16)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETE":
        raise RuntimeError(f"baseline artifact is not complete: {path} ({value.get('status')})")
    return value


def metric_rows(name: str, path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for prefix in PREFIXES:
        aggregate = payload["aggregate"][str(prefix)]
        standard = aggregate["standard_mean"]
        persistent = aggregate["persistent_mean"]
        rows.append({
            "method": name,
            "prefix": prefix,
            "standard_old_acc": float(standard["old_acc"]),
            "standard_new_acc": float(standard["new_acc"]),
            "standard_h_score": float(standard["h_score"]),
            "standard_all_acc": float(standard["all_acc"]),
            "persistent_commit_ct": float(persistent["commit_ct"]),
            "persistent_false_assignment_rate": float(persistent["false_assignment_rate"]),
            "comparable": True,
            "status": "COMPLETE",
            "source": str(path.resolve()),
            "source_sha256": sha256_file(path),
        })
    return rows


def main() -> int:
    out = ensure_output_layout()
    files = {
        "nearest": OUTPUT_TARGET / "tables/gt_nearest.json",
        "dpmeans": OUTPUT_TARGET / "tables/gt_dpmeans.json",
        "phe": OUTPUT_TARGET / "tables/gt_phe.json",
    }
    payloads = {name: read_json(path) for name, path in files.items()}
    rows = [row for name, path in files.items() for row in metric_rows(name, path, payloads[name])]
    geometry_path = OUTPUT_TARGET / "audit/geometry_audit.json"
    geometry = read_json(geometry_path)
    current_path = OUTPUT_TARGET / "audit/current_model_contract.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    if current.get("decision") != "STRUCTURALLY_LOADABLE_BUT_NOT_COMPARABLE":
        raise RuntimeError("legacy current model contract unexpectedly became comparable; require explicit protocol review")

    best_by_prefix = {}
    for prefix in PREFIXES:
        candidates = [row for row in rows if row["prefix"] == prefix]
        best = max(candidates, key=lambda row: row["standard_h_score"])
        best_by_prefix[str(prefix)] = {"method": best["method"], "standard_h_score": best["standard_h_score"], "persistent_commit_ct": best["persistent_commit_ct"]}

    csv_buffer = io.StringIO()
    fields = ["method", "prefix", "standard_old_acc", "standard_new_acc", "standard_h_score", "standard_all_acc", "persistent_commit_ct", "persistent_false_assignment_rate", "comparable", "status", "source"]
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in rows)
    atomic_write_text(out / "tables/gt_benchmark_table.csv", csv_buffer.getvalue())
    result = {
        "schema_version": "trackocd.v2.gt_benchmark_table.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "GT-track Val, four fixed stream orders, one global Hungarian Standard OCD mapping, full-stream Persistent OCD",
        "prefixes": list(PREFIXES),
        "comparable_methods": ["nearest", "dpmeans", "phe"],
        "excluded_current_model": {
            "method": "H3_REPRO",
            "status": current.get("decision"),
            "contract_artifact": str(current_path.resolve()),
            "contract_artifact_sha256": sha256_file(current_path),
            "reason": "legacy 0.8*CLS+0.2*ROI and Phase19R-normalized geometry do not match v2 pure-CLS plus v2 geometry contract",
            "known_role_coverage": current.get("known_role_coverage"),
        },
        "rows": rows,
        "best_by_prefix": best_by_prefix,
        "geometry_audit": {
            "source": str(geometry_path.resolve()),
            "source_sha256": sha256_file(geometry_path),
            "status": geometry.get("status"),
            "splits": {split: {"tracks": value.get("tracks"), "categories": value.get("categories"), "videos": value.get("videos")} for split, value in geometry.get("splits", {}).items()},
        },
        "test_semantic_accessed": False,
        "model_input_gt_labels": False,
    }
    atomic_json(out / "tables/gt_benchmark_table.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
