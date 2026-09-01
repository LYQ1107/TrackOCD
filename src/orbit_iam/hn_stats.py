"""Generate positive/negative similarity statistics from training logs."""
from __future__ import annotations

import argparse
import csv
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.stats)))
    out_rows = []
    for r in rows:
        own = r["own_exists"] == "1"
        out_rows.append({
            "epoch": r["epoch"],
            "first": r["first"],
            "mem_size": r["mem_size"],
            "positive_sim": r["best_sim"] if own else "",
            "hard_negative_sim_mean": r["hard_neg_sim_mean"],
            "n_hard_neg": r["n_hard_neg"],
        })
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", args.out, len(out_rows))


if __name__ == "__main__":
    main()
