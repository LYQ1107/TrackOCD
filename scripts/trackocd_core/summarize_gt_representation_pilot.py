#!/usr/bin/env python3
"""Reformat existing aggregate results only; no rerun, image or model access."""
import csv
import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_write_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--r1', action='store_true')
    r1 = parser.parse_args().r1
    prefix = 'gt_representation_r1' if r1 else 'gt_representation'
    receipt = json.loads((ROOT / f"outputs/trackocd_core/audit/{prefix}_evaluation.json").read_text())
    rows = receipt["per_seed_order_aggregates"]
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    filename = 'GT_REPRESENTATION_R1_PILOT_RESULTS.csv' if r1 else 'GT_REPRESENTATION_PILOT_RESULTS.csv'
    atomic_write_text(ROOT / 'outputs/trackocd_core/audit' / filename, text.getvalue())
    print(json.dumps({"existing_aggregate_rows": len(rows), "experiments_rerun": False}))
