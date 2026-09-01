#!/usr/bin/env python3
"""Extract per-iteration training stats from OVTR train.log into CSV."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pat = re.compile(
        r"Epoch: \[(\d+)\]\s+\[\s*(\d+)/\d+\].*?"
        r"lr: ([0-9.]+).*?grad_norm: ([0-9.]+).*?"
        r"loss: ([0-9.]+) \(([0-9.]+)\)")
    tco_pat = re.compile(r"frame_1_loss_tco: ([0-9.]+) \(([0-9.]+)\)")
    rows = []
    for line in Path(args.log).read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        row = {
            "epoch": int(m.group(1)),
            "iter": int(m.group(2)),
            "lr": float(m.group(3)),
            "grad_norm": float(m.group(4)),
            "loss": float(m.group(5)),
            "loss_avg": float(m.group(6)),
        }
        t = tco_pat.search(line)
        if t:
            row["tco_loss"] = float(t.group(1))
            row["tco_loss_avg"] = float(t.group(2))
        rows.append(row)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
