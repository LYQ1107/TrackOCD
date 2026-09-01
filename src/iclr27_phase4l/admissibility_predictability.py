"""Phase 4L admissibility predictability audit (Root Cause A).

Simple diagnostic models only: standardized logistic regression and a
depth-3 tree, 5-fold stratified CV.  GT enters only as the offline
label (valid object evidence vs FP).  Feature sets are compared:
detector-score-only, tracking-only, combined.
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

DET_FEATURES = [
    "det_score", "bbox_area", "bbox_aspect", "mask_area_frac",
    "mask_bbox_ratio", "appearance_norm", "dino_norm", "p_known",
    "best_known", "known_margin", "best_novel", "assoc_ap_score",
    "assoc_fn_score", "assoc_sem_delta", "assoc_assigned", "track_age",
    "novel_support",
]
DETECTOR_ONLY = ["det_score"]
TRACK_DET = [f for f in DET_FEATURES if f != "det_score"]

TRACK_FEATURES = [
    "length", "max_age", "max_gap", "consecutive_max", "mean_det_score",
    "max_det_score", "mean_p_known", "std_p_known", "mean_best_known",
    "mean_known_margin", "appearance_prefix_cos",
    "appearance_prefix_min_cos", "mean_bbox_iou", "mean_scale_change",
    "mean_aspect_change", "semantic_switch_rate", "gid_switch_rate",
    "mean_assoc_ap",
]
TRACK_DETECTOR_ONLY = ["mean_det_score", "max_det_score"]
TRACK_TRACKING = [f for f in TRACK_FEATURES
                  if f not in TRACK_DETECTOR_ONLY]


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


def evaluate(X, y, label, rows_out, feats, tag, scope):
    y = np.asarray(y, dtype=int)
    if len(set(y)) < 2 or len(y) < 10:
        rows_out.append({
            "tag": tag, "scope": scope, "features": label,
            "n": len(y), "n_positive": int(y.sum()),
            "auroc": "", "auprc": "", "tree_auroc": "",
        })
        return
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    logi = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, C=1.0))
    yp = cross_val_predict(logi, X, y, cv=skf, method="predict_proba")[:, 1]
    auroc = float(roc_auc_score(y, yp))
    auprc = float(average_precision_score(y, yp))
    tree = DecisionTreeClassifier(max_depth=3, random_state=11)
    yt = cross_val_predict(tree, X, y, cv=skf, method="predict_proba")[:, 1]
    tree_auroc = float(roc_auc_score(y, yt))
    rows_out.append({
        "tag": tag, "scope": scope, "features": label,
        "n": len(y), "n_positive": int(y.sum()),
        "auroc": round(auroc, 4), "auprc": round(auprc, 4),
        "tree_auroc": round(tree_auroc, 4),
    })


def main():
    det = load("admissibility_detection_features.csv")
    trk = load("admissibility_tracklet_features.csv")
    out_rows = []
    feat_rows = []

    # ---- detection level ----
    det = [r for r in det if all(k in r for k in DET_FEATURES)]
    Xd = {f: [num(r[f]) for r in det] for f in DET_FEATURES}
    yd = [1 if r["gt_role"] in ("known", "novel") else 0 for r in det]
    scope = "detection_all"
    for label, feats in (("detector_only", DETECTOR_ONLY),
                         ("tracking_only", TRACK_DET),
                         ("combined", DET_FEATURES)):
        X = np.column_stack([Xd[f] for f in feats])
        evaluate(X, yd, label, out_rows, feats, "j1b", scope)
        for i, f in enumerate(feats):
            try:
                evaluate(X[:, [i]], yd, "single:" + f, out_rows,
                         [f], "j1b", scope)
            except ValueError:
                pass
    # time-conditioned (track age buckets)
    buckets = [(1, 1, "age1"), (2, 2, "age2"), (3, 4, "age3_4"),
               (5, 8, "age5_8"), (9, 16, "age9_16"), (17, 10 ** 9,
                                                      "age17plus")]
    for lo, hi, name in buckets:
        idx = [i for i, r in enumerate(det) if lo <= int(r["track_age"])
               <= hi]
        if len(idx) < 10:
            continue
        X = np.column_stack([[Xd[f][i] for i in idx] for f in DET_FEATURES])
        y = [yd[i] for i in idx]
        evaluate(X, y, "combined", out_rows, DET_FEATURES, "j1b",
                 "detection_" + name)

    # ---- tracklet level ----
    trk = [r for r in trk if all(k in r for k in TRACK_FEATURES)]
    Xt = {f: [num(r[f]) for r in trk] for f in TRACK_FEATURES}
    yt = [1 if r["role"] == "tp" else 0 for r in trk]
    for label, feats in (("detector_only", TRACK_DETECTOR_ONLY),
                         ("tracking_only", TRACK_TRACKING),
                         ("combined", TRACK_FEATURES)):
        X = np.column_stack([Xt[f] for f in feats])
        evaluate(X, yt, label, out_rows, feats, "j1b", "tracklet_all")
        for f in feats:
            try:
                evaluate(X[:, [feats.index(f)]], yt, "single:" + f,
                         out_rows, [f], "j1b", "tracklet_all")
            except ValueError:
                pass
    # persistent FP only (length >= 6)
    idx = [i for i, r in enumerate(trk) if int(r["length"]) >= 6]
    if len(idx) >= 10:
        X = np.column_stack([[Xt[f][i] for i in idx]
                             for f in TRACK_FEATURES])
        y = [yt[i] for i in idx]
        evaluate(X, y, "combined", out_rows, TRACK_FEATURES, "j1b",
                 "tracklet_persistent_fp")
    # tracklet length buckets
    for lo, hi, name in buckets:
        idx = [i for i, r in enumerate(trk) if lo <= int(r["length"])
               <= hi]
        if len(idx) < 10:
            continue
        X = np.column_stack([[Xt[f][i] for i in idx]
                             for f in TRACK_FEATURES])
        y = [yt[i] for i in idx]
        evaluate(X, y, "combined", out_rows, TRACK_FEATURES, "j1b",
                 "tracklet_" + name)

    with open(OUT / "admissibility_predictability.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(json.dumps(out_rows, indent=1))
    print("ADMISSIBILITY_PREDICTABILITY_DONE")


if __name__ == "__main__":
    main()
