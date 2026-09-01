"""Phase 4G routing-bias conditional audit.

For ORBIT-MDC M2 novel tracks (official + long), conditions Novel->Known on
visual evidence (best-known similarity, known margin) and then measures the
residual effect of the current memory state.  A simple logistic probe
quantifies whether memory state adds independent explanation.  GT role is
offline audit only; nothing here is used to train or tune the method.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict

import numpy as np

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"


def load_novel(stream):
    rows = list(csv.DictReader(open(
        f"{ROOT}/outputs/iclr27_phase4f/audit/memory_trajectory_m2_{stream}.csv")))
    out = []
    for r in rows:
        if r["true_role"] != "novel":
            continue
        out.append({
            "arrival": int(r["arrival_index"]),
            "best_known_sim": float(r["known_best_sim"]),
            "known_margin": float(r["known_margin"]),
            "gate_prob": float(r["gate_prob"]),
            "n2k": int(r["predicted_action"] == "KNOWN"),
            "memory_size": int(r["memory_size"]),
            "mean_support": float(r["mean_support"]),
            "p50_support": float(r["p50_support"]),
            "p90_support": float(r["p90_support"]),
            "low_support_count": int(r["low_conf_count"]),
            "mean_dispersion": float(r["mean_dispersion"]),
            "mean_conf": float(r["mean_conf"]),
            "hub_count": int(r["hub_count_offline"]),
            "known_origin_count": int(r["known_origin_count_offline"]),
            "bucket": r["memory_bucket"],
        })
    return out


def bucket_of(m):
    if m < 33:
        return "0-32"
    if m < 129:
        return "33-128"
    if m < 257:
        return "129-256"
    return "257+"


def conditional_table(rows, key, bins, out_path):
    table = []
    for lo, hi, label in bins:
        sel = [r for r in rows if lo <= r[key] < hi]
        if not sel:
            continue
        by_bucket = defaultdict(list)
        for r in sel:
            by_bucket[bucket_of(r["memory_size"])].append(r)
        for b in ["0-32", "33-128", "129-256", "257+"]:
            g = by_bucket.get(b, [])
            if not g:
                continue
            table.append({
                "visual_bin": label, "memory_bucket": b,
                "n": len(g),
                "novel_to_known_rate": sum(r["n2k"] for r in g) / len(g),
                "mean_gate_prob": float(np.mean([r["gate_prob"] for r in g])),
            })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    return table


def probe(rows, out_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.inspection import permutation_importance
    X = np.array([[r["best_known_sim"], r["known_margin"]]
                  for r in rows], dtype=np.float32)
    y = np.array([r["n2k"] for r in rows])
    mem_feats = ["log_mem", "mean_support", "low_support_ratio",
                 "mean_dispersion", "mean_conf", "hub_count"]
    Xm = np.array([[
        math.log1p(r["memory_size"]),
        r["mean_support"],
        r["low_support_count"] / max(r["memory_size"], 1),
        r["mean_dispersion"],
        r["mean_conf"],
        r["hub_count"] / max(r["memory_size"], 1),
    ] for r in rows], dtype=np.float32)
    Xfull = np.concatenate([X, Xm], axis=1)
    rows_out = []
    for name, XX, feats in [("visual_only", X, ["best_known_sim", "known_margin"]),
                            ("visual+memory", Xfull,
                             ["best_known_sim", "known_margin"] + mem_feats),
                            ("memory_only", Xm, mem_feats)]:
        if len(np.unique(y)) < 2:
            rows_out.append({"model": name, "auc": float("nan"),
                             "log_loss": float("nan"), "coef": "", "note": "constant y"})
            continue
        clf = LogisticRegression(max_iter=2000)
        clf.fit(XX, y)
        pred = clf.predict_proba(XX)[:, 1]
        auc = roc_auc_score(y, pred)
        ll = log_loss(y, pred)
        rows_out.append({
            "model": name, "auc": round(auc, 4),
            "log_loss": round(ll, 4),
            "coef": "; ".join(f"{f}={c:.3f}" for f, c in zip(feats, clf.coef_[0])),
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    return rows_out


def main():
    import pathlib
    out = pathlib.Path(f"{ROOT}/outputs/iclr27_phase4g/audit")
    out.mkdir(parents=True, exist_ok=True)
    sim_bins = [(0.2, 0.4, "0.2-0.4"), (0.4, 0.5, "0.4-0.5"),
                (0.5, 0.6, "0.5-0.6"), (0.6, 0.7, "0.6-0.7"),
                (0.7, 1.01, "0.7+")]
    mar_bins = [(-0.001, 0.02, "<0.02"), (0.02, 0.05, "0.02-0.05"),
                (0.05, 0.1, "0.05-0.1"), (0.1, 1.01, "0.1+")]
    for stream in ["official", "long"]:
        rows = load_novel(stream)
        print("==", stream, "novel tracks", len(rows))
        t1 = conditional_table(rows, "best_known_sim", sim_bins,
                               f"{out}/routing_bias_by_similarity_{stream}.csv")
        t2 = conditional_table(rows, "known_margin", mar_bins,
                               f"{out}/routing_bias_by_margin_{stream}.csv")
        # memory-state rows: N2K/gate by bucket (overall)
        t3 = []
        for b in ["0-32", "33-128", "129-256", "257+"]:
            g = [r for r in rows if bucket_of(r["memory_size"]) == b]
            if g:
                t3.append({"memory_bucket": b, "n": len(g),
                           "novel_to_known_rate": sum(r["n2k"] for r in g) / len(g),
                           "mean_gate_prob": float(np.mean([r["gate_prob"] for r in g]))})
        with open(f"{out}/routing_bias_by_memory_state_{stream}.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(t3[0].keys()))
            w.writeheader()
            w.writerows(t3)
        p = probe(rows, f"{out}/routing_bias_probe_{stream}.csv")
        for r in p:
            print(r)
        print("by_sim (bucket rows):", len(t1), "by_margin:", len(t2))


if __name__ == "__main__":
    main()
