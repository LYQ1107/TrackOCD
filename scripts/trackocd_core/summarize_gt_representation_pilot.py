#!/usr/bin/env python3
"""Reformat existing aggregate results only; no rerun, image or model access."""
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_write_text


if __name__ == "__main__":
    receipt = json.loads((ROOT / "outputs/trackocd_core/audit/gt_representation_evaluation.json").read_text())
    rows = receipt["per_seed_order_aggregates"]
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(ROOT / "outputs/trackocd_core/audit/GT_REPRESENTATION_PILOT_RESULTS.csv", text.getvalue())
    print(json.dumps({"existing_aggregate_rows": len(rows), "experiments_rerun": False}))
