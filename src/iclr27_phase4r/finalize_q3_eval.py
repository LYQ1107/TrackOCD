#!/usr/bin/env python3
"""Merge Phase 4R Q3 pilot eval with Q0/Q1/Q2 and Q2-alpha controls.

Parses the TETA summary block from each eval log with the exact column
layout used by the TETer fork, then joins it with the frozen proposal
metrics emitted by ovtr_main_eval.py.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

# TETA50 header printed by the eval: TETA LocS AssocS ClsS LocRe LocPr
# AssocRe AssocPr ClsRe ClsPr
TETA_COLS = [
    "teta", "locs", "assocs", "clss",
    "locre", "locpr", "assocre", "assocpr", "clsre", "clspr",
]

PROPOSAL_KEYS = [
    "total_rows", "total_novel", "total_known", "total_fp",
    "total_persistent_fp", "persistent_fp_per_frame",
    "novel_recall_at_fp_1.0", "early_age0_recall_at_fp_1.0",
    "fp_per_frame_at_recall_0.3", "reject_persistent_fp_at_fp_1.0",
]


def parse_teta_block(log_path: Path):
    """Return {combined/base/novel: {col: float}} from an eval log."""
    text = log_path.read_text(errors="replace")
    out = {}
    lines = [ln.split() for ln in text.splitlines()]
    for tokens in lines:
        if not tokens:
            continue
        head = tokens[0].lower()
        if head in ("combined", "base", "novel") and len(tokens) >= 11:
            nums = tokens[1:11]
            if all(re.fullmatch(r"-?[\d.]+", x) for x in nums):
                out[head] = dict(zip(TETA_COLS, map(float, nums)))
    return out


def load_proposals(prefix: str):
    p = Path(f"{prefix}_metrics.json")
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return {k: data["dev"].get(k) for k in PROPOSAL_KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--model", action="append", default=[],
        metavar="NAME=EVAL_LOG=PROPOSAL_PREFIX")
    args = ap.parse_args()

    rows = {}
    for spec in args.model:
        name, log_s, prefix = spec.split("=", 2)
        teta = parse_teta_block(Path(log_s))
        props = load_proposals(prefix)
        rows[name] = {
            "teta": teta.get("combined"),
            "teta_base": teta.get("base"),
            "teta_novel": teta.get("novel"),
            "proposals_dev": props,
        }

    out = {"models": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))

    # Compact review table.
    def f(v, nd=3):
        return "NA" if v is None else f"{float(v):.{nd}f}"

    print(f"{'model':<8} {'proposals':>9} {'persFP/f':>9} "
          f"{'novel@1FP':>9} {'age0@1FP':>9} {'FP/f@r.3':>9} "
          f"{'TETA':>7} {'LocRe':>7} {'AssocA':>7} {'novTETA':>8}")
    for name, r in rows.items():
        p = r["proposals_dev"] or {}
        t = r["teta"] or {}
        tn = r["teta_novel"] or {}
        print(f"{name:<8} {f(p.get('total_rows'), 0):>9} "
              f"{f(p.get('persistent_fp_per_frame')):>9} "
              f"{f(p.get('novel_recall_at_fp_1.0')):>9} "
              f"{f(p.get('early_age0_recall_at_fp_1.0')):>9} "
              f"{f(p.get('fp_per_frame_at_recall_0.3')):>9} "
              f"{f(t.get('teta')):>7} {f(t.get('locre')):>7} "
              f"{f(t.get('assocs')):>7} {f(tn.get('teta')):>8}")
    print("FINALIZE_Q3_EVAL_DONE")


if __name__ == "__main__":
    main()
