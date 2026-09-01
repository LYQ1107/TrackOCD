#!/usr/bin/env python3
"""Aggregate all experiment JSONs into summary CSV and summary.json."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_jsons(glob_pattern):
    rows = []
    for p in sorted(PROJECT_ROOT.glob(glob_pattern)):
        rows.append(json.loads(p.read_text()))
    return rows


def main():
    out_dir = PROJECT_ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    # B0/B1 kmeans
    for d in load_jsons("runs/gt_kmeans/*.json"):
        all_rows.append(
            {
                "method": d["method"],
                "encoder": d["encoder"],
                "subset": d["subset"],
                "seed": "oracle",
                "acc_all": d["acc_all"],
                "acc_known": d["acc_known"],
                "acc_novel": d["acc_novel"],
                "nmi": d["nmi"],
                "ari": d["ari"],
                "predicted_categories": d["predicted_categories"],
                "category_count_abs_error": d["category_count_abs_error"],
                "mean_fragmentation": d["mean_fragmentation"],
                "merge_error": d["merge_error"],
                "duplicate_creation_rate": d["duplicate_creation_rate"],
                "mean_assignment_delay": d.get("mean_assignment_delay"),
            }
        )

    # B2 NCM
    for d in load_jsons("runs/gt_online_ncm/*.json"):
        all_rows.append(
            {
                "method": "online_ncm",
                "encoder": d["encoder"],
                "subset": d["subset"],
                "seed": d["seed"],
                "acc_all": d["acc_all"],
                "acc_known": d["acc_known"],
                "acc_novel": d["acc_novel"],
                "nmi": d["nmi"],
                "ari": d["ari"],
                "predicted_categories": d["predicted_categories"],
                "category_count_abs_error": d["category_count_abs_error"],
                "mean_fragmentation": d["mean_fragmentation"],
                "merge_error": d["merge_error"],
                "duplicate_creation_rate": d["duplicate_creation_rate"],
                "mean_assignment_delay": d.get("mean_assignment_delay"),
            }
        )

    # B3 PHE-Track GT
    for d in load_jsons("runs/gt_phe_track/*.json"):
        all_rows.append(
            {
                "method": "phe_track_gt",
                "encoder": d["encoder"],
                "subset": d["subset"],
                "seed": f"{d['seed']}_train{d['encoder']}_seed{d.get('seed')}",
                "acc_all": d["acc_all"],
                "acc_known": d["acc_known"],
                "acc_novel": d["acc_novel"],
                "nmi": d["nmi"],
                "ari": d["ari"],
                "predicted_categories": d["predicted_categories"],
                "category_count_abs_error": d["category_count_abs_error"],
                "mean_fragmentation": d["mean_fragmentation"],
                "merge_error": d["merge_error"],
                "duplicate_creation_rate": d["duplicate_creation_rate"],
                "mean_assignment_delay": d.get("mean_assignment_delay"),
            }
        )

    # B4 PHE-Track predicted
    for d in load_jsons("runs/pred_phe_track/*.json"):
        all_rows.append(
            {
                "method": "phe_track_pred",
                "encoder": d["encoder"],
                "subset": "full",
                "seed": f"seed{d['seed']}",
                "acc_all": d["acc_all"],
                "acc_known": d["acc_known"],
                "acc_novel": d["acc_novel"],
                "nmi": d["nmi"],
                "ari": d["ari"],
                "predicted_categories": d["predicted_categories"],
                "category_count_abs_error": d["category_count_abs_error"],
                "mean_fragmentation": d["mean_fragmentation"],
                "merge_error": d["merge_error"],
                "duplicate_creation_rate": d["duplicate_creation_rate"],
                "mean_assignment_delay": d.get("mean_assignment_delay"),
            }
        )

    fieldnames = [
        "method", "encoder", "subset", "seed", "acc_all", "acc_known", "acc_novel",
        "nmi", "ari", "predicted_categories", "category_count_abs_error",
        "mean_fragmentation", "merge_error", "duplicate_creation_rate",
        "mean_assignment_delay",
    ]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # Rewrite per-method CSVs cleanly from JSON artifacts
    method_files = {
        "gt_kmeans.csv": [r for r in all_rows if r["method"].startswith("kmeans")],
        "gt_online_ncm.csv": [r for r in all_rows if r["method"] == "online_ncm"],
        "gt_phe_track.csv": [r for r in all_rows if r["method"] == "phe_track_gt"],
        "pred_phe_track.csv": [r for r in all_rows if r["method"] == "phe_track_pred"],
    }
    for name, rows in method_files.items():
        with open(out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # seed mean/std for B3 main stream
    phe_main = [d for d in load_jsons("runs/gt_phe_track/*main.json")]
    agg = {}
    for d in phe_main:
        key = (d["encoder"], d["subset"])
        agg.setdefault(key, []).append(d)
    summary = {}
    for (enc, subset), ds in agg.items():
        stats_ = {}
        for k in ["acc_all", "acc_known", "acc_novel", "nmi", "ari"]:
            vals = [d[k] for d in ds]
            stats_[k] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "values": vals,
            }
        summary[f"{enc}_{subset}_phe_main"] = stats_
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
