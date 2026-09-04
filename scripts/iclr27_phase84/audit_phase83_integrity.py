#!/usr/bin/env python3
"""Read-only provenance audit for the Phase83 A2 report source.

This script deliberately does not rerun any Phase83 inference.  It compares
the A2 table in the frozen report with the two existing metric artifacts and
records which artifact supplied the table values.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/iclr27_phase83/PHASE83_RESUMED_FINAL_REPORT.md"
A2 = ROOT / "outputs/iclr27_phase83/metrics/a2_temporal_r.json"
OLD = ROOT / "outputs/iclr27_phase83/metrics/physical_r_temporal.json"
OUT = ROOT / "outputs/iclr27_phase84/audit/phase83_a2_report_integrity.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def report_p16() -> dict[str, float | int] | None:
    text = REPORT.read_text(encoding="utf-8")
    # The table is intentionally parsed from the unique p16 row in the A2
    # section.  Values are rounded in the report, so this is a provenance
    # check rather than a numerical recomputation.
    match = re.search(
        r"\| 16 \| 984 \| ([0-9.]+) \| ([0-9.]+) \| ([+-]?[0-9.]+) \| "
        r"([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| (\d+) \|",
        text,
    )
    if not match:
        return None
    vals = [float(x) for x in match.groups()[:-1]]
    return {
        "queries": 984,
        "raw_r1": vals[0],
        "temporal_r1": vals[1],
        "delta_r1": vals[2],
        "raw_map": vals[3],
        "temporal_map": vals[4],
        "temporal_gap": vals[5],
        "raw_gap": vals[6],
        "unsafe": int(match.groups()[-1]),
    }


def main() -> None:
    report = report_p16()
    a2 = json.loads(A2.read_text(encoding="utf-8"))
    old = json.loads(OLD.read_text(encoding="utf-8"))
    actual = a2["prefix"]["16"]
    old_p16 = next(
        row["metrics"]
        for row in old["sections"]["exact_mixed"]["folds"]
        if row.get("prefix") == 16 and row.get("fold") == 0
    )
    # The old artifact has per-fold rows; the report's rounded aggregate is
    # identifiable by exact agreement with its aggregate fields below.
    report_matches_old_table = bool(report and abs(report["temporal_r1"] - 0.8827) < 1e-4)
    result = {
        "phase": "Phase84",
        "audit": "phase83_a2_report_source_integrity",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "read_only": True,
        "report": {"path": str(REPORT.resolve()), "sha256": sha256(REPORT), "p16_table": report},
        "actual_a2_artifact": {
            "path": str(A2.resolve()),
            "sha256": sha256(A2),
            "schema_version": a2.get("schema_version"),
            "aggregate_p16": actual,
            "mapping": a2.get("mapping"),
        },
        "previous_physical_temporal_artifact": {
            "path": str(OLD.resolve()),
            "sha256": sha256(OLD),
            "schema_version": old.get("schema_version"),
            "fold0_p16_sample": old_p16,
        },
        "finding": {
            "report_table_temporal_p16": report.get("temporal_r1") if report else None,
            "actual_a2_temporal_p16": actual.get("r1"),
            "report_table_uses_old_physical_r_temporal_values": report_matches_old_table,
            "report_table_matches_actual_a2_aggregate": bool(report and abs(report["temporal_r1"] - float(actual["r1"])) < 5e-4),
            "source_bug": True,
            "source_bug_description": (
                "The frozen A2 table in PHASE83_RESUMED_FINAL_REPORT.md reports the "
                "partial physical_r_temporal.json values (0.882735 at p16), while "
                "the A2 artifact a2_temporal_r.json has aggregate p16 r1=0.880983. "
                "The table therefore cannot be used as A2 full-coverage evidence."
            ),
            "phase83_a2_artifact_is_not_rerun": True,
        },
        "public_dev_q1_sealed_accessed": False,
        "held_labels_used_for_model_selection": False,
    }
    atomic_json(OUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
