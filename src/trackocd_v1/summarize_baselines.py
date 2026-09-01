#!/usr/bin/env python3
"""Summarize corrected baselines and build the legacy-vs-corrected table."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "outputs" / "trackocd_v1" / "metrics"
SEED_STREAMS = ["main_seed1027", "main_seed1028", "main_seed1029"]

KEY_METRICS = [
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
        w.writerows(rows)


def fnum(row, key):
    try:
        v = row.get(key)
        return float(v) if v not in (None, "") else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def summary():
    rows = []
    for proto in ("pure", "ov_assisted"):
        data = read_csv(OUT / f"{proto}_baselines.csv")
        for method in sorted({r["method"] for r in data}):
            for subset in sorted({r["subset"] for r in data}):
                seeds = [r for r in data if r["method"] == method and r["subset"] == subset and r["seed"] in SEED_STREAMS]
                if not seeds:
                    continue
                row = {"protocol": proto, "method": method, "subset": subset}
                for k in KEY_METRICS:
                    vals = [fnum(r, k) for r in seeds]
                    vals = [v for v in vals if v == v]
                    row[f"{k}_mean"] = statistics.mean(vals) if vals else ""
                    row[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
                rows.append(row)
    write_csv(OUT / "corrected_baseline_summary.csv", rows)
    print("summary rows", len(rows))


def legacy_json_map():
    m = {}
    m[("B0", "main")] = PROJECT_ROOT / "runs" / "gt_kmeans" / "dinov2_single_full_assignments.json"
    m[("B1", "main")] = PROJECT_ROOT / "runs" / "gt_kmeans" / "dinov2_mean_full_assignments.json"
    for st in ("main", "main_seed1027", "main_seed1028", "main_seed1029"):
        m[("B2", st)] = PROJECT_ROOT / "runs" / "gt_online_ncm" / f"dinov2_full_{st}.json"
    m[("B3", "main")] = PROJECT_ROOT / "runs" / "gt_phe_track" / "dinov2_seed1027_full_main.json"
    m[("B3", "main_seed1027")] = PROJECT_ROOT / "runs" / "gt_phe_track" / "dinov2_seed1027_full_main_seed1027.json"
    m[("B3", "main_seed1028")] = PROJECT_ROOT / "runs" / "gt_phe_track" / "dinov2_seed1028_full_main_seed1028.json"
    m[("B3", "main_seed1029")] = PROJECT_ROOT / "runs" / "gt_phe_track" / "dinov2_seed1029_full_main_seed1029.json"
    for st in ("main", "main_seed1027", "main_seed1028", "main_seed1029"):
        m[("B4", st)] = PROJECT_ROOT / "runs" / "arch1_5" / f"ocd_v2_dual_{st}_full.json"
    return m


def legacy_vs_corrected():
    legacy = legacy_json_map()
    rows = []
    legacy_pairs = [
        ("acc_all", "all_track_acc"),
        ("acc_known", "overall_known_acc"),
        ("acc_novel", "route_aware_novel_acc"),
        ("nmi", "novel_only_nmi"),
        ("ari", "novel_only_ari"),
        ("category_count_abs_error", "novel_count_abs_error"),
    ]
    for proto in ("pure", "ov_assisted"):
        data = read_csv(OUT / f"{proto}_baselines.csv")
        for method, st in legacy:
            p = legacy[(method, st)]
            if not p.exists():
                continue
            lj = json.loads(p.read_text())
            corr = next(
                (r for r in data if r["method"] == method and r["seed"] == st and r["subset"] == "full"),
                None,
            )
            if corr is None:
                continue
            for lk, ck in legacy_pairs:
                rows.append({
                    "protocol": proto, "method": method, "subset": "full", "seed": st,
                    "legacy_evaluator": "legacy", "metric": lk,
                    "legacy_value": lj.get(lk),
                    "corrected_metric": ck,
                    "corrected_value": corr.get(ck),
                })
    write_csv(OUT / "legacy_vs_corrected.csv", rows)
    print("legacy-vs-corrected rows", len(rows))


if __name__ == "__main__":
    summary()
    legacy_vs_corrected()
