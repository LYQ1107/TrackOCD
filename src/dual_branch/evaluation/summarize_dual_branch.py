#!/usr/bin/env python3
"""Aggregate D0-D3 dual-branch runs into the required CSVs and gate check."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

RUNS = PROJECT_ROOT / "runs" / "dual_branch"
OUT = PROJECT_ROOT / "outputs" / "dual_branch" / "metrics"
SEEDS = ["main_seed1027", "main_seed1028", "main_seed1029"]
METHODS = ["D0", "D1", "D2", "D3T", "D3D"]
KEYS = [
    "all_track_acc", "overall_known_acc", "supported_known_acc",
    "zero_shot_known_acc", "known_to_novel_error", "known_misclassification_rate",
    "novel_routing_recall", "novel_routing_precision",
    "false_known_absorption_rate", "unresolved_novel_rate",
    "route_aware_novel_acc", "conditional_novel_acc", "novel_only_nmi",
    "novel_only_ari", "macro_novel_class_acc", "predicted_novel_count",
    "novel_count_abs_error", "mean_fragmentation", "merge_error",
    "duplicate_creation_rate", "mean_assignment_delay",
    "macro_known_novel_harmonic",
]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = []
    for proto in ("pure", "ov_assisted"):
        for subset in ("full", "repeated", "balanced"):
            for method in METHODS:
                rows = []
                for seed in SEEDS:
                    p = RUNS / f"{method}_{proto}_{subset}_{seed}.json"
                    if not p.exists():
                        continue
                    r = json.loads(p.read_text())
                    row = {
                        "method": method, "protocol": proto, "subset": subset,
                        "seed": seed,
                        **{k: r[k] for k in KEYS},
                    }
                    rows.append(row)
                    all_rows.append(row)
                if not rows:
                    continue
                srow = {"method": method, "protocol": proto, "subset": subset}
                for k in KEYS:
                    vals = [float(r[k]) for r in rows if r.get(k) is not None]
                    if not vals:
                        continue
                    srow[f"{k}_mean"] = statistics.mean(vals)
                    srow[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
                summary.append(srow)

    def dump(name, methods):
        rows = [r for r in all_rows if r["method"] in methods]
        if rows:
            write_csv(OUT / name, rows)

    dump("d0_reproduction.csv", ["D0"])
    dump("d1_reproduction.csv", ["D1"])
    dump("d2_hard_dual_branch.csv", ["D2"])
    dump("d3_oracle_route.csv", ["D3T", "D3D"])
    write_csv(OUT / "final_summary.csv", summary)
    write_csv(OUT / "d0_d1_d2_d3_all_runs.csv", all_rows)

    # gate check: D2 vs frozen gates on pure full
    d2 = next(r for r in summary if r["method"] == "D2" and r["protocol"] == "pure" and r["subset"] == "full")
    d1 = next(r for r in summary if r["method"] == "D1" and r["protocol"] == "pure" and r["subset"] == "full")
    d0 = next(r for r in summary if r["method"] == "D0" and r["protocol"] == "pure" and r["subset"] == "full")
    checks = {
        "known_acc_ge_0.80": d2["overall_known_acc_mean"] >= 0.80,
        "route_novel_ge_0.286": d2["route_aware_novel_acc_mean"] >= 0.286,
        "cond_novel_ge_0.63": d2["conditional_novel_acc_mean"] >= 0.63,
        "nmi_ge_0.89": d2["novel_only_nmi_mean"] >= 0.89,
        "ari_ge_0.48": d2["novel_only_ari_mean"] >= 0.48,
        "count_error_le_100": d2["novel_count_abs_error_mean"] <= 100,
        "novel_count_not_30pct_worse_than_d0": (
            d2["predicted_novel_count_mean"] <= 1.3 * d0["predicted_novel_count_mean"]
        ),
        "repeated_or_balanced_not_both_below_d0": not (
            next(r for r in summary if r["method"] == "D2" and r["protocol"] == "pure" and r["subset"] == "repeated")["route_aware_novel_acc_mean"] < d0["route_aware_novel_acc_mean"]
            and next(r for r in summary if r["method"] == "D2" and r["protocol"] == "pure" and r["subset"] == "balanced")["route_aware_novel_acc_mean"] < d0["route_aware_novel_acc_mean"]
        ),
    }
    # D1/D2 route mask and known consistency from paired diagnostic
    paired = list(csv.DictReader(open(OUT / "paired_route_diagnostics.csv")))
    masks_ok = all(r["route_mask_identical"] == "True" for r in paired)
    known_ok = abs(d2["overall_known_acc_mean"] - d1["overall_known_acc_mean"]) < 1e-9
    checks["d1_d2_route_masks_identical"] = masks_ok
    checks["d1_d2_known_identical"] = known_ok
    passed = all(checks.values())
    status = "PASS_DUAL_BRANCH" if passed else "FAIL_DUAL_BRANCH"
    # PARTIAL definition: known preserved and cond novel/NMI/ARI recover vs D1
    partial = (
        known_ok
        and d2["conditional_novel_acc_mean"] > d1["conditional_novel_acc_mean"]
        and d2["novel_only_nmi_mean"] > d1["novel_only_nmi_mean"]
        and d2["novel_only_ari_mean"] > d1["novel_only_ari_mean"]
    )
    if partial and not passed:
        status = "PARTIAL_DUAL_BRANCH"
    result = {
        "status": status,
        "passed": passed,
        "partial": partial,
        "checks": checks,
        "d0": {k: d0[f"{k}_mean"] for k in ("overall_known_acc", "route_aware_novel_acc", "conditional_novel_acc", "novel_only_nmi", "novel_only_ari", "predicted_novel_count", "novel_count_abs_error")},
        "d1": {k: d1[f"{k}_mean"] for k in ("overall_known_acc", "route_aware_novel_acc", "conditional_novel_acc", "novel_only_nmi", "novel_only_ari", "predicted_novel_count", "novel_count_abs_error")},
        "d2": {k: d2[f"{k}_mean"] for k in ("overall_known_acc", "route_aware_novel_acc", "conditional_novel_acc", "novel_only_nmi", "novel_only_ari", "predicted_novel_count", "novel_count_abs_error")},
    }
    (RUNS / "dual_branch_gate.json").write_text(json.dumps(result, indent=2))
    (RUNS / "status.txt").write_text(status + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
