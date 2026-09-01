#!/usr/bin/env python3
"""Generate required router artifacts from existing results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = PROJECT_ROOT / "outputs" / "domain_router"
RUNS = PROJECT_ROOT / "runs" / "domain_router"
DATA = PROJECT_ROOT / "data" / "domain_router" / "proxy_protocol"


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    rows = list(csv.DictReader(open(OUT / "metrics" / "router_full_results.csv")))
    r0 = [r for r in rows if r["router"] == "R0"]
    write_csv(OUT / "metrics" / "r0_reproduction.csv", r0)
    # current proxy analysis (legacy class-split proxy diagnostics)
    outer = list(csv.DictReader(open(OUT / "metrics" / "proxy_outer_folds.csv")))
    rows_out = []
    for r in outer:
        if r["router"] == "R0":
            rows_out.append({
                "target_domain": r["target_domain"], "fold_index": r["fold_index"],
                "r0_hmean": r["hmean"], "r0_known_recall": r["known_recall"],
                "r0_novel_recall": r["novel_recall"], "r0_threshold": r["threshold"],
            })
    write_csv(OUT / "audit" / "current_proxy_analysis.csv", rows_out)
    # hashes dir
    hashes = DATA / "hashes"
    hashes.mkdir(parents=True, exist_ok=True)
    import hashlib
    for p in sorted((DATA / "folds").glob("*.json")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        (hashes / (p.stem + ".sha256")).write_text(h + "\n")
    print("finalized")


if __name__ == "__main__":
    main()
