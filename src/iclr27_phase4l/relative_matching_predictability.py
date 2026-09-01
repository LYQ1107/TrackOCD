"""Phase 4L relative novel matching predictability audit (Root Cause B).

Target: for true novel queries that are not same-track continuations,
can causal geometry separate SAME_NOVEL (correct EXISTING_NOVEL) from
DIFFERENT_NOVEL (should create NEW_NOVEL)?  Simple models, 5-fold CV.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4l" / "audit"

FEATURES = [
    "best_cos", "second_cos", "margin", "novel_minus_known",
    "support_causal", "member_count", "member_mean_cos",
    "member_std_cos", "member_min_cos", "member_max_cos",
    "query_zscore", "proto_distinct_tracks", "local_entropy",
    "near_count",
]
ABSOLUTE = ["best_cos"]


def load(name):
    p = OUT / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def run(X, y, label, out):
    y = np.asarray(y, dtype=int)
    if len(set(y)) < 2 or len(y) < 10:
        out.append({
            "features": label, "n": len(y),
            "n_same_novel": int(y.sum()), "n_different_novel": len(y) -
            int(y.sum()), "auroc": "", "auprc": "", "tree_auroc": ""})
        return
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    logi = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, C=1.0))
    yp = cross_val_predict(logi, X, y, cv=skf, method="predict_proba")[:, 1]
    auroc = float(roc_auc_score(y, yp))
    auprc = float(average_precision_score(y, yp))
    tree = DecisionTreeClassifier(max_depth=3, random_state=13)
    yt = cross_val_predict(tree, X, y, cv=skf, method="predict_proba")[:, 1]
    tree_auroc = float(roc_auc_score(y, yt))
    out.append({
        "features": label, "n": len(y),
        "n_same_novel": int(y.sum()), "n_different_novel": len(y) -
        int(y.sum()), "auroc": round(auroc, 4), "auprc": round(auprc, 4),
        "tree_auroc": round(tree_auroc, 4),
    })


def main():
    pairs = load("novel_matching_pairs.csv")
    pairs = [r for r in pairs if all(k in r for k in FEATURES)]
    out = []
    # true novel, cross-track only (the real EXISTING vs NEW decision)
    sub = [r for r in pairs if r["det_gt_role"] == "novel" and
           r["same_track"] == "0"]
    X = {f: [num(r[f]) for r in sub] for f in FEATURES}
    y = [1 if r["case"] == "SAME_NOVEL" else 0 for r in sub]
    for label, feats in (("absolute_best_cos", ABSOLUTE),
                         ("relative_full", FEATURES)):
        run(np.column_stack([X[f] for f in feats]), y, label, out)
        for f in feats:
            run(np.column_stack([X[f]]), y, "single:" + f, out)
    # all true novel queries including same-track continuations
    sub2 = [r for r in pairs if r["det_gt_role"] == "novel"]
    X2 = {f: [num(r[f]) for r in sub2] for f in FEATURES}
    y2 = [1 if r["case"] == "SAME_NOVEL" else 0 for r in sub2]
    run(np.column_stack([X2[f] for f in FEATURES]), y2, "relative_all", out)
    # known collisions vs FP stream diagnostics
    diag = {
        "n_total": len(pairs),
        "n_same_novel": sum(1 for r in pairs if r["case"] == "SAME_NOVEL"),
        "n_different_novel": sum(
            1 for r in pairs if r["case"] == "DIFFERENT_NOVEL"),
        "n_known_collision": sum(
            1 for r in pairs if r["case"] == "KNOWN_COLLISION"),
        "n_fp_query": sum(1 for r in pairs if r["case"] == "FP_QUERY"),
    }
    with open(OUT / "relative_matching_predictability.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    (OUT / "matching_diagnostics.json").write_text(
        json.dumps(diag, indent=1))
    print(json.dumps(diag, indent=1))
    print(json.dumps(out, indent=1))
    print("RELATIVE_MATCHING_PREDICTABILITY_DONE")


if __name__ == "__main__":
    main()
