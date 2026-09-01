#!/usr/bin/env python3
"""Paired route diagnostics: on the identical routed-novel subset (D1 == D2
route mask), compare transformer-embedding B2 vs DINO-mean B2 discovery."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
from src.trackocd_v1.rerun_baselines import load_gt, subset_ids

RUNS = PROJECT_ROOT / "runs" / "dual_branch"
OUT = PROJECT_ROOT / "outputs" / "dual_branch" / "metrics"
STREAMS = ("main", "main_seed1027", "main_seed1028", "main_seed1029")
SUBSETS = ("full", "repeated", "balanced")


def main():
    rows = []
    for proto in ("pure", "ov_assisted"):
        gt = load_gt(proto)
        for subset in SUBSETS:
            for stream in STREAMS:
                p1 = RUNS / f"D1_{proto}_{subset}_{stream}.json"
                p2 = RUNS / f"D2_{proto}_{subset}_{stream}.json"
                if not p1.exists() or not p2.exists():
                    continue
                r1 = json.loads(p1.read_text())
                r2 = json.loads(p2.read_text())
                log1 = {p["sample_id"]: p for p in r1["prediction_log"]}
                log2 = {p["sample_id"]: p for p in r2["prediction_log"]}
                assert set(log1) == set(log2)
                sub = subset_ids(proto, subset)
                routed = [
                    sid for sid in log1
                    if sid in sub and log1[sid]["prediction_type"] == "novel"
                ]
                # routing masks must be identical
                mismatch = [
                    sid for sid in routed
                    if log2[sid]["prediction_type"] != "novel"
                ]
                if mismatch:
                    raise SystemExit(f"route mask mismatch {proto}/{subset}/{stream}: {len(mismatch)}")
                ev = TrackOCDEvaluator(gt)
                res1 = ev.evaluate(
                    [log1[s] for s in routed], subset_ids=set(routed))
                res2 = ev.evaluate(
                    [log2[s] for s in routed], subset_ids=set(routed))
                row = {
                    "protocol": proto, "subset": subset, "seed": stream,
                    "routed_novel_tracks": len(routed),
                    "route_mask_identical": len(mismatch) == 0,
                    "D1_cond_novel_acc": res1["conditional_novel_acc"],
                    "D2_cond_novel_acc": res2["conditional_novel_acc"],
                    "D1_novel_nmi": res1["novel_only_nmi"],
                    "D2_novel_nmi": res2["novel_only_nmi"],
                    "D1_novel_ari": res1["novel_only_ari"],
                    "D2_novel_ari": res2["novel_only_ari"],
                    "D1_predicted_novel_count": res1["predicted_novel_count"],
                    "D2_predicted_novel_count": res2["predicted_novel_count"],
                    "D1_fragmentation": res1["mean_fragmentation"],
                    "D2_fragmentation": res2["mean_fragmentation"],
                    "D1_merge_error": res1["merge_error"],
                    "D2_merge_error": res2["merge_error"],
                }
                rows.append(row)
                print(row, flush=True)
    if rows:
        with open(OUT / "paired_route_diagnostics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("done")


if __name__ == "__main__":
    main()
