#!/usr/bin/env python3
"""Aggregate Architecture 1.5 Stage A per-run CSVs into:
- stage_a_summary.csv        (3-seed mean/std per method x gate x subset)
- {method}.csv               (all gate/subset/seed rows, required file names)
- stage_a_gate.json          (gate evaluation vs reproduced B2)
- learned_gate_gap.json      (learned vs oracle gate comparison)
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

csv.field_size_limit(1 << 30)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
csv.field_size_limit(10**9)

OUT = PROJECT_ROOT / "outputs" / "arch1_5" / "metrics"
RUNS = PROJECT_ROOT / "runs" / "arch1_5"
SEEDS = ["main_seed1027", "main_seed1028", "main_seed1029"]
METHODS = ["spherical_kmeans", "dpmeans", "candidate_buffer", "ocd_v2"]
GATES = ["clip", "dino", "dual", "dual_lr"]
SUBSETS = ["full", "repeated", "balanced"]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean_std(values):
    vals = [float(v) for v in values]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def numeric(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def load_b2():
    rows = read_csv(OUT / "b2_reproduced.csv")
    seed_rows = [r for r in rows if r["seed"] in SEEDS and r["subset"] == "full"]
    b2 = {}
    for k in (
        "acc_all", "acc_known", "acc_novel", "nmi", "ari",
        "predicted_categories", "category_count_abs_error",
        "mean_fragmentation", "merge_error", "duplicate_creation_rate",
    ):
        m, s = mean_std([numeric(r, k) for r in seed_rows])
        b2[k] = {"mean": m, "std": s, "values": [numeric(r, k) for r in seed_rows]}
    return b2


def summarize():
    b2 = load_b2()
    summary_rows = []
    for method in METHODS:
        all_rows = []
        for gate in GATES:
            src = OUT / f"{method}_{gate}.csv"
            if not src.exists():
                continue
            rows = read_csv(src)
            all_rows.extend(rows)
            for subset in SUBSETS:
                seed_rows = [r for r in rows if r["seed"] in SEEDS and r["subset"] == subset]
                if not seed_rows:
                    continue
                row = {"method": method, "gate": gate, "subset": subset}
                for k in (
                    "acc_all", "acc_known", "acc_novel", "nmi", "ari",
                    "predicted_categories", "category_count_abs_error",
                    "mean_fragmentation", "merge_error", "duplicate_creation_rate",
                ):
                    m, s = mean_std([numeric(r, k) for r in seed_rows])
                    row[f"{k}_mean"] = m
                    row[f"{k}_std"] = s
                    row[f"{k}_values"] = json.dumps([numeric(r, k) for r in seed_rows])
                mem = [r for r in seed_rows if r.get("memory_stats")]
                if mem:
                    try:
                        ms = json.loads(mem[0]["memory_stats"]) if isinstance(mem[0]["memory_stats"], str) else mem[0]["memory_stats"]
                        row["memory_stats_sample"] = json.dumps(ms)
                    except Exception:
                        pass
                summary_rows.append(row)
        if all_rows:
            write_csv(OUT / f"{method}.csv", all_rows)
    write_csv(OUT / "stage_a_summary.csv", summary_rows)
    return b2, summary_rows


def gate_check(b2, summary_rows):
    main = [r for r in summary_rows if r["method"] == "ocd_v2" and r["gate"] == "dual_lr"]
    if not main:
        return {"passed": False, "reason": "no ocd_v2/dual rows"}
    by_subset = {r["subset"]: r for r in main}
    full = by_subset.get("full")
    repeated = by_subset.get("repeated")
    balanced = by_subset.get("balanced")
    if full is None or repeated is None or balanced is None:
        return {"passed": False, "reason": "missing subset rows"}

    b2_full = b2["acc_novel"]["mean"]
    b2_known = b2["acc_known"]["mean"]
    b2_nmi = b2["nmi"]["mean"]
    b2_ari = b2["ari"]["mean"]
    b2_count = b2["category_count_abs_error"]["mean"]

    checks = {}
    checks["novel_acc_gain"] = full["acc_novel_mean"] - b2_full
    checks["novel_acc_gain_ok"] = checks["novel_acc_gain"] >= 0.03
    checks["count_error"] = full["category_count_abs_error_mean"]
    checks["count_error_target"] = min(b2_count / 2.0, 135.0)
    checks["count_error_ok"] = checks["count_error"] <= checks["count_error_target"]
    checks["nmi_delta"] = full["nmi_mean"] - b2_nmi
    checks["nmi_ok"] = checks["nmi_delta"] >= -0.01
    checks["ari_delta"] = full["ari_mean"] - b2_ari
    checks["ari_ok"] = checks["ari_delta"] >= 0.0
    checks["known_acc_delta"] = full["acc_known_mean"] - b2_known
    checks["known_acc_ok"] = checks["known_acc_delta"] >= -0.02
    checks["repeated_novel_delta"] = repeated["acc_novel_mean"] - b2_full
    checks["repeated_novel_ok"] = repeated["acc_novel_mean"] >= b2_full
    checks["balanced_novel_delta"] = balanced["acc_novel_mean"] - b2_full
    checks["balanced_novel_ok"] = balanced["acc_novel_mean"] >= b2_full
    seed_vals = json.loads(full["acc_novel_values"])
    checks["novel_seed_min"] = min(seed_vals)
    checks["novel_seed_collapse_ok"] = checks["novel_seed_min"] >= b2_full - 0.05
    checks["all_ok"] = all(
        checks[k]
        for k in (
            "novel_acc_gain_ok", "count_error_ok", "nmi_ok", "ari_ok",
            "known_acc_ok", "repeated_novel_ok", "balanced_novel_ok",
            "novel_seed_collapse_ok",
        )
    )
    checks["b2_reference"] = {
        "novel_acc_mean": b2_full,
        "known_acc_mean": b2_known,
        "nmi_mean": b2_nmi,
        "ari_mean": b2_ari,
        "count_error_mean": b2_count,
    }
    result = {
        "main_method": "ocd_v2",
        "main_gate": "dual_lr",
        "passed": checks["all_ok"],
        "checks": checks,
        "status": "PASS_STAGE_A" if checks["all_ok"] else "FAIL_STAGE_A",
    }
    (RUNS / "stage_a_gate.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def learned_oracle_gap(summary_rows):
    learned_path = OUT / "learned_gate.csv"
    oracle_path = OUT / "oracle_gate.csv"
    if not learned_path.exists() or not oracle_path.exists():
        return None
    learned = read_csv(learned_path)
    oracle = read_csv(oracle_path)
    oracle_full = [r for r in oracle if r["subset"] == "full" and r["seed"] in SEEDS]
    gap = {}
    if oracle_full:
        for k in ("acc_all", "acc_novel", "nmi", "ari", "predicted_categories", "category_count_abs_error"):
            m, s = mean_std([numeric(r, k) for r in oracle_full])
            gap[f"oracle_{k}_mean"] = m
            gap[f"oracle_{k}_std"] = s
    main = [r for r in summary_rows if r["method"] == "ocd_v2" and r["gate"] == "dual_lr" and r["subset"] == "full"]
    if main and oracle_full:
        gap["learned_vs_oracle_novel_gap"] = main[0]["acc_novel_mean"] - gap["oracle_acc_novel_mean"]
        gap["learned_vs_oracle_nmi_gap"] = main[0]["nmi_mean"] - gap["oracle_nmi_mean"]
    (RUNS / "learned_gate_gap.json").write_text(json.dumps(gap, indent=2, default=str))
    return gap


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    b2, summary_rows = summarize()
    result = gate_check(b2, summary_rows)
    gap = learned_oracle_gap(summary_rows)
    print(json.dumps(result, indent=2, default=str))
    if gap:
        print("gap", json.dumps(gap, indent=2))
