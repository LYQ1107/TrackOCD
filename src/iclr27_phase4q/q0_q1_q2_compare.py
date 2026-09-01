#!/usr/bin/env python3
"""Aggregate Q0/Q1/Q2 (and Phase 4P P0/P2/P1+) into one comparison table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

KEYS = [
    ("novel_recall_at_fp_1.0", "Novel Recall @1FP"),
    ("novel_recall_at_fp_3.0", "Novel Recall @3FP"),
    ("novel_recall_at_fp_5.0", "Novel Recall @5FP"),
    ("fp_per_frame_at_recall_0.3", "FP/frame @r0.3"),
    ("persistent_fp_per_frame", "persistent FP/frame"),
    ("early_age0_recall_at_fp_1.0", "early age0 @1FP"),
    ("early_age1_recall_at_fp_1.0", "early age1 @1FP"),
]


def load_metrics(prefix):
    p = Path(prefix)
    if p.is_file():
        return json.loads(p.read_text())
    mp = Path(str(prefix) + "_metrics.json")
    if mp.is_file():
        return json.loads(mp.read_text())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    sources = {
        "P0": ROOT / "outputs/iclr27_phase4p/ovtr_main/p0_official/proposals_metrics.json",
        "P2": ROOT / "outputs/iclr27_phase4p/ovtr_main/p2_tco_epoch1/proposals_metrics.json",
        "P1+onP0": ROOT / "outputs/iclr27_phase4q/p1plus/on_p0/p1plus_report.json",
        "P1+onP2": ROOT / "outputs/iclr27_phase4q/p1plus/on_p2/p1plus_report.json",
        "P1+onQ0": ROOT / "outputs/iclr27_phase4q/p1plus/on_q0_long/p1plus_report.json",
        "P1+onQ1": ROOT / "outputs/iclr27_phase4q/p1plus/on_q1_long/p1plus_report.json",
        "P1+onQ2": ROOT / "outputs/iclr27_phase4q/p1plus/on_q2_long/p1plus_report.json",
        "Q0": ROOT / "outputs/iclr27_phase4q/q0_long/proposals_metrics.json",
        "Q1": ROOT / "outputs/iclr27_phase4q/q1_long/proposals_metrics.json",
        "Q2": ROOT / "outputs/iclr27_phase4q/q2_long/proposals_metrics.json",
    }

    table = {}
    for name, path in sources.items():
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        dev = data.get("dev", data)
        ho = data.get("heldout", data)
        table[name] = {
            "dev": {k: dev.get(k) for k, _ in KEYS},
            "heldout": {k: ho.get(k) for k, _ in KEYS},
        }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(table, indent=2, default=str))

    print(f"{'model':<10} {'split':<8}", end="")
    for _, label in KEYS:
        print(f"{label:>24}", end="")
    print()
    for name, d in table.items():
        for split in ("dev", "heldout"):
            print(f"{name:<10} {split:<8}", end="")
            for k, _ in KEYS:
                v = d[split].get(k)
                s = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                print(f"{s:>24}", end="")
            print()
    print("Q0_Q1_Q2_COMPARISON_DONE")


if __name__ == "__main__":
    main()
