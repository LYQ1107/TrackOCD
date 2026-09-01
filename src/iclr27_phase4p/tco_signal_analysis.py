"""Phase 4P Q2-style mechanism analysis: learned trajectory-conditioned
objectness (LR on dev) vs frame-online track-confirmation heuristics.

Evaluation is on the held-out persistent subset (rows whose physical
track already has >=2 prior frames).  All LR features are frame-online.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from iclr27_phase4p.trajectory_objectness_audit import (
    CAUSAL_FEATS, STATIC_FEATS, load_pop, role_of,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4p" / "dev"


def mat(rows, fp_mode="persistent_FP"):
    cols = STATIC_FEATS + CAUSAL_FEATS
    X, y = [], []
    for r in rows:
        role = role_of(r)
        if role == "NOVEL":
            ok = True
        elif role == "FP":
            ok = fp_mode == "all_FP" or r["prior_hits"] >= 2
        else:
            ok = False
        if ok:
            X.append([r[c] for c in cols])
            y.append(1 if role == "NOVEL" else 0)
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    med = np.nanmedian(X, axis=0)
    med[np.isnan(med)] = 0.0
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = med[j]
    return X, y


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    dev = load_pop(
        ROOT / "outputs" / "iclr27_phase4p" / "audit" /
        "detection_population_dev_fixed.csv", "dev")
    ho = load_pop(
        ROOT / "outputs" / "iclr27_phase4p" / "audit" /
        "detection_population_heldout_corrected_fixed.csv", "heldout")
    Xd, yd = mat(dev)
    Xh, yh = mat(ho)
    novel_total = int(yh.sum())
    fp_total = int((yh == 0).sum())
    sc = StandardScaler().fit(Xd)
    clf = LogisticRegression(max_iter=3000, C=0.1).fit(
        sc.transform(Xd), yd)
    s = clf.decision_function(sc.transform(Xh))
    order = np.argsort(-s)
    ys = yh[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(ys == 0)
    rec = tp / max(novel_total, 1)
    rows = []
    for target in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        idx = int(np.searchsorted(rec, target))
        if idx >= len(rec):
            continue
        rows.append({
            "mode": "heldout_persistent", "method": "learned_TCO_LR",
            "novel_recall": round(float(rec[idx]), 4),
            "kept": idx + 1, "novel": int(tp[idx]), "fp": int(fp[idx]),
            "precision": round(float(tp[idx] / (idx + 1)), 4),
            "fp_kept_frac": round(float(fp[idx] / fp_total), 4),
        })
    # Heuristics on the same persistent subset.
    for k in (1, 2, 4):
        for s_th in (0.5, 0.6, 0.7):
            keep = [r for r in ho
                    if role_of(r) in ("NOVEL", "FP")
                    and r["prior_hits"] >= k and r["score"] >= s_th]
            nv = sum(1 for r in keep if role_of(r) == "NOVEL")
            fp = sum(1 for r in keep if role_of(r) == "FP")
            rows.append({
                "mode": "heldout_persistent",
                "method": f"heuristic_age{k}_score{s_th}",
                "novel_recall": round(nv / max(novel_total, 1), 4),
                "kept": len(keep), "novel": nv, "fp": fp,
                "precision": round(nv / max(len(keep), 1), 4),
                "fp_kept_frac": round(fp / max(fp_total, 1), 4),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tco_mechanism_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("TCO_MECHANISM_ANALYSIS_DONE")
    for r in rows:
        print(r["method"], "rec", r["novel_recall"], "prec",
              r["precision"], "fp_frac", r["fp_kept_frac"])


if __name__ == "__main__":
    main()
