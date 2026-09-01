"""Rebuild prototype-confidence audit CSVs from validated audit logs.

The per-prototype statistics and correlations are computed from the frozen
replay logs (audit_log_<model>_<stream>.json), which reproduce the official
and long-stream frozen metrics exactly. GT is used only for the offline
purity/risk columns.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"


def per_prototype_rows(logs):
    by_vid = defaultdict(list)
    for l in logs:
        vid = l.get("predicted_virtual_novel_id")
        if vid is not None:
            by_vid[int(vid)].append(l)
    rows = []
    for vid, ls in by_vid.items():
        cls = Counter(int(l["class"]) for l in ls if l["role"] == "novel")
        n = max(sum(cls.values()), 1)
        primary = max(cls, key=cls.get) if cls else None
        purity = cls[primary] / n if primary is not None else 0.0
        wrong = sum(1 for l in ls
                    if l["predicted_action"] == "EXISTING_NOVEL"
                    and l.get("assigned_primary_class") is not None
                    and int(l["assigned_primary_class"]) != int(l["class"]))
        first_merge = sum(1 for l in ls
                          if l["first_occurrence"]
                          and l["predicted_action"] == "EXISTING_NOVEL")
        last = ls[-1]
        def num(k):
            v = last.get(k)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")
        rows.append({
            "virtual_id": vid,
            "support": sum(1 for l in ls if l["role"] == "novel"),
            "radius": num("assigned_radius"),
            "dispersion": num("assigned_dispersion"),
            "mean_margin": num("assigned_mean_margin"),
            "min_margin": num("assigned_min_margin"),
            "low_margin_count": num("assigned_low_margin_count"),
            "recent_stability": num("assigned_recent_stability"),
            "age": num("assigned_age"),
            "conf_legal": num("assigned_conf_legal"),
            "distinct_classes": len(cls),
            "purity_offline": purity,
            "wrong_existing_caused": wrong,
            "first_merge_caused": first_merge,
            "primary_class_offline": primary,
        })
    return rows


def main():
    all_rows = []
    for model in ["c1", "c2"]:
        for stream in ["long", "official"]:
            logs = json.load(open(
                f"{ROOT}/outputs/iclr27_phase4e/audit/audit_log_{model}_{stream}.json"))
            for r in per_prototype_rows(logs):
                r["model"] = model.upper()
                r["stream"] = stream
                all_rows.append(r)
    fn = list(all_rows[0].keys())
    with open(f"{ROOT}/outputs/iclr27_phase4e/audit/prototype_confidence_analysis.csv",
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(all_rows)

    feats = ["support", "dispersion", "mean_margin", "min_margin",
             "low_margin_count", "recent_stability", "age", "radius",
             "conf_legal"]
    targets = {"purity_offline": "purity",
               "wrong_existing_caused": "n_wrong_assignments",
               "first_merge_caused": "n_false_merges"}
    corr = []
    for model in ["C1", "C2"]:
        for stream in ["long", "official"]:
            ps = [r for r in all_rows if r["model"] == model
                  and r["stream"] == stream]
            for feat in feats:
                for t, tname in targets.items():
                    xs, ys = [], []
                    for r in ps:
                        x, y = r[feat], r[t]
                        if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                           and math.isfinite(x) and math.isfinite(y):
                            xs.append(x)
                            ys.append(y)
                    if len(xs) >= 5 and len(set(xs)) > 1 and len(set(ys)) > 1:
                        pr = pearsonr(xs, ys)
                        sr = spearmanr(xs, ys)
                        corr.append({"model": model, "stream": stream,
                                     "feature": feat, "target": tname,
                                     "pearson_r": float(pr[0]),
                                     "pearson_p": float(pr[1]),
                                     "spearman_r": float(sr[0]),
                                     "spearman_p": float(sr[1]),
                                     "n": len(xs)})
    cfn = list(corr[0].keys())
    with open(f"{ROOT}/outputs/iclr27_phase4e/audit/confidence_purity_correlation.csv",
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cfn)
        w.writeheader()
        w.writerows(corr)
    print("prototypes", len(all_rows), "corr rows", len(corr))
    for r in corr:
        if r["model"] == "C1" and r["stream"] == "official" and r["feature"] in (
                "support", "recent_stability", "conf_legal") and r["target"] in (
                    "purity", "n_wrong_assignments"):
            print(r["feature"], r["target"], round(r["pearson_r"], 3),
                  round(r["spearman_r"], 3), r["n"])


if __name__ == "__main__":
    main()
