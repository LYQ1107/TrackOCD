#!/usr/bin/env python3
"""Rebuild {method}_{gate}.csv from per-run JSONs in runs/arch1_5.
This repairs CSVs that only contain the last subset due to an overwrite bug
in earlier run_stage_a.py versions, without re-running experiments."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS = PROJECT_ROOT / "runs" / "arch1_5"
OUT = PROJECT_ROOT / "outputs" / "arch1_5" / "metrics"

METHODS = ["spherical_kmeans", "dpmeans", "candidate_buffer", "ocd_v2"]
GATES = ["clip", "dino", "dual"]
SUBSETS = ["full", "repeated", "balanced"]
SEEDS = ["main", "main_seed1027", "main_seed1028", "main_seed1029"]


def main():
    n = 0
    for method in METHODS:
        for gate in GATES:
            rows = []
            for subset in SUBSETS:
                for seed in SEEDS:
                    p = RUNS / f"{method}_{gate}_{seed}_{subset}.json"
                    if not p.exists():
                        print(f"MISSING {p.name}")
                        continue
                    r = json.loads(p.read_text())
                    rows.append(r)
            if rows:
                fieldnames = list(rows[0].keys())
                with open(OUT / f"{method}_{gate}.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    w.writeheader()
                    for r in rows:
                        w.writerow(r)
                n += len(rows)
                print(f"{method}_{gate}.csv rows={len(rows)}")
    print("total rows", n)


if __name__ == "__main__":
    main()
