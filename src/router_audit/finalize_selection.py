#!/usr/bin/env python3
"""Build selection_reconstruction.csv from OOF scores and feasibility."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = PROJECT_ROOT / "outputs/router_audit"
RUNS = PROJECT_ROOT / "runs/router_audit"


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    oof = json.loads((RUNS / "oof_scores.json").read_text())
    feas = list(csv.DictReader(open(OUT / "fold_feasibility.csv")))
    sel = list(csv.DictReader(open(OUT.parent / "domain_router/metrics/router_selection.csv")))
    r0_hm = max(float(r["outer_hmean_mean"]) for r in sel if r["router"] == "R0")
    rows = []
    old_outer = list(csv.DictReader(open(OUT.parent / "domain_router/metrics/proxy_outer_folds.csv")))
    for r in old_outer:
        if r["router"] == "R0":
            rows.append({
                "target_domain": r["target_domain"], "seed": 1027, "method": "R0",
                "proxy_known": "", "proxy_novel": "",
                "known_recall": r["known_recall"], "novel_recall": r["novel_recall"],
                "hmean": r["hmean"], "auroc": "", "aupr": "",
                "threshold": r["threshold"], "known_recall_floor": "",
                "feasible": "True", "selected": "no",
                "exclusion_reason": "reference baseline",
            })
    for name in ("R0", "R1", "R2", "R3", "R4"):
        domains = {}
        for r in oof.get(name, []):
            domains.setdefault(r["target_domain"], []).append(r)
        for f in feas:
            if f["method"] != name:
                continue
            td = f["target_domain"]
            rows_d = domains.get(td, [])
            scores = np.array([r["score"] for r in rows_d])
            labels = np.array([r["label"] for r in rows_d])
            auroc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.0
            aupr = average_precision_score(labels, scores) if len(set(labels)) > 1 else 0.0
            selected = "yes" if (name != "R0" and float(f["hmean"]) > r0_hm and f["feasible"] == "True") else "no"
            rows.append({
                "target_domain": td, "seed": 1027, "method": name,
                "proxy_known": f["proxy_known"], "proxy_novel": f["proxy_novel"],
                "known_recall": f["known_recall"], "novel_recall": f["novel_recall"],
                "hmean": f["hmean"], "auroc": round(auroc, 4), "aupr": round(aupr, 4),
                "threshold": f["threshold"], "known_recall_floor": f["known_recall_floor"],
                "feasible": f["feasible"], "selected": selected,
                "exclusion_reason": "" if f["feasible"] == "True" else "no threshold meets known-recall floor",
            })
    write_csv(OUT / "selection_reconstruction.csv", rows)
    print("selection_reconstruction rows", len(rows))


if __name__ == "__main__":
    main()
