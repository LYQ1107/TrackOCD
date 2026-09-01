"""Phase 4P minimal experiment 2: full-stream trajectory-conditioned
objectness gate on the FIXED detection populations.

Question: can a frame-online learned gate (LR over current-frame +
causal-prefix features) lift novel recall at fixed FP/frame from the
corrected D0 numbers (dev 0.049 / heldout 0.022 @1 FP/frame), on the
WHOLE stream (all ages, not just persistent subset)?

Evaluation follows the exact Phase 4O protocol:
  - rows ranked by gate score (descending);
  - FP/frame = cumulative FP / total frames of the split;
  - known rows count for known/valid recall but not for FP/frame;
  - dev curve uses 5-fold out-of-fold scores (honest in-domain);
  - heldout curve uses a model trained on full dev (transfer).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from iclr27_phase4p.trajectory_objectness_audit import (
    CAUSAL_FEATS, STATIC_FEATS, load_pop, role_of,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4p" / "audit"

DEV_POP = OUT / "detection_population_dev_fixed.csv"
HO_POP = OUT / "detection_population_heldout_corrected_fixed.csv"
HELDOUT_TAO = ROOT / "outputs" / "iclr27_phase4n" / "audit" / \
    "validation_heldout_tao_corrected.json"
REPLAY_DIR = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "replay_packages"


def n_frames(mode):
    if mode == "dev":
        return sum(len(list(p.glob("frame_*.npz")))
                   for p in REPLAY_DIR.iterdir())
    return len({im["id"] for im in
                json.loads(HELDOUT_TAO.read_text())["images"]})


def mat_rows(rows, cols, positive):
    X, y, idx = [], [], []
    for i, r in enumerate(rows):
        role = role_of(r)
        if positive == "valid":
            if role in ("NOVEL", "KNOWN"):
                ok = True
            elif role == "FP":
                ok = False
            else:
                continue
        else:
            if role == "NOVEL":
                ok = True
            elif role == "FP":
                ok = False
            else:
                continue
        X.append([r[c] for c in cols])
        y.append(1 if ok else 0)
        idx.append(i)
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    med = np.nanmedian(X, axis=0)
    med[np.isnan(med)] = 0.0
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = med[j]
    return np.asarray(X, float), np.asarray(y, int), idx


def all_rows_X(rows, cols):
    X = np.asarray([[r[c] for c in cols] for r in rows], float)
    med = np.nanmedian(X, axis=0)
    med[np.isnan(med)] = 0.0
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = med[j]
    return X


def oof_scores(rows, cols, positive, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    X, y, idx = mat_rows(rows, cols, positive)
    idx = np.asarray(idx)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    s = np.zeros(len(rows))
    covered = np.zeros(len(rows), dtype=bool)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=0.1).fit(
            sc.transform(X[tr]), y[tr])
        Xall = all_rows_X(rows, cols)
        s_all = clf.decision_function(sc.transform(Xall))
        s[idx[te]] = s_all[idx[te]]
        covered[idx[te]] = True
    # rows not covered by any fold (e.g. KNOWN rows in a NOVEL-vs-FP task)
    # get scores from one model trained on all trainable rows.
    if not covered.all():
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=3000, C=0.1).fit(
            sc.transform(X), y)
        s_all = clf.decision_function(sc.transform(all_rows_X(rows, cols)))
        s[~covered] = s_all[~covered]
    return s


def transfer_scores(dev, ho, cols, positive):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xd, yd, _ = mat_rows(dev, cols, positive)
    sc = StandardScaler().fit(Xd)
    clf = LogisticRegression(max_iter=3000, C=0.1).fit(
        sc.transform(Xd), yd)
    return clf.decision_function(sc.transform(all_rows_X(ho, cols)))


def eval_curve(rows, scores, nfr):
    roles = np.asarray([role_of(r) for r in rows])
    order = np.argsort(-np.asarray(scores, float))
    n_novel = int((roles == "NOVEL").sum())
    n_known = int((roles == "KNOWN").sum())
    n_fp = int((roles == "FP").sum())
    cum_novel = np.cumsum(roles[order] == "NOVEL")
    cum_known = np.cumsum(roles[order] == "KNOWN")
    cum_fp = np.cumsum(roles[order] == "FP")
    return {
        "order": order, "roles": roles,
        "novel_recall": cum_novel / max(n_novel, 1),
        "known_recall": cum_known / max(n_known, 1),
        "valid_recall": (cum_novel + cum_known) / max(n_novel + n_known, 1),
        "fp_per_frame": cum_fp / max(nfr, 1),
        "n_novel": n_novel, "n_known": n_known, "n_fp": n_fp,
    }


def rec_at_fp(ev, target):
    idx = np.where(ev["fp_per_frame"] <= target)[0]
    if len(idx) == 0:
        return None, None, None
    i = idx[-1]
    return (float(ev["novel_recall"][i]), float(ev["known_recall"][i]),
            float(ev["valid_recall"][i]))


def fp_at_rec(ev, target):
    idx = np.where(ev["novel_recall"] >= target)[0]
    if len(idx) == 0:
        return None
    return float(ev["fp_per_frame"][idx[0]])


def main():
    dev = load_pop(DEV_POP, "dev")
    ho = load_pop(HO_POP, "heldout")
    nfr = {"dev": n_frames("dev"), "heldout": n_frames("heldout")}
    print("frames:", nfr)
    print("rows:", len(dev), len(ho))
    print("novel/known/fp dev:",
          sum(1 for r in dev if role_of(r) == "NOVEL"),
          sum(1 for r in dev if role_of(r) == "KNOWN"),
          sum(1 for r in dev if role_of(r) == "FP"))
    print("novel/known/fp ho:",
          sum(1 for r in ho if role_of(r) == "NOVEL"),
          sum(1 for r in ho if role_of(r) == "KNOWN"),
          sum(1 for r in ho if role_of(r) == "FP"))

    variants = [
        ("score_only", None, "raw"),
        ("static_novel", STATIC_FEATS, "novel"),
        ("causal_novel", CAUSAL_FEATS, "novel"),
        ("all_novel", STATIC_FEATS + CAUSAL_FEATS, "novel"),
        ("all_valid", STATIC_FEATS + CAUSAL_FEATS, "valid"),
    ]
    curves = []
    oper = []
    for name, cols, positive in variants:
        for mode, rows in (("dev", dev), ("heldout", ho)):
            if positive == "raw":
                s = [r["score"] for r in rows]
            elif mode == "dev":
                s = oof_scores(rows, cols, positive)
            else:
                s = transfer_scores(dev, ho, cols, positive)
            ev = eval_curve(rows, s, nfr[mode])
            for i in range(0, len(rows), max(1, len(rows) // 400)):
                curves.append({
                    "method": name, "mode": mode,
                    "fp_per_frame": round(float(ev["fp_per_frame"][i]), 5),
                    "novel_recall": round(float(ev["novel_recall"][i]), 5),
                    "known_recall": round(float(ev["known_recall"][i]), 5),
                    "valid_recall": round(float(ev["valid_recall"][i]), 5),
                })
            for fp_t in (0.1, 0.3, 1.0, 3.0, 10.0):
                nr, kr, vr = rec_at_fp(ev, fp_t)
                oper.append({
                    "method": name, "mode": mode, "fp_per_frame": fp_t,
                    "novel_recall": None if nr is None else round(nr, 4),
                    "known_recall": None if kr is None else round(kr, 4),
                    "valid_recall": None if vr is None else round(vr, 4),
                })
            for rec_t in (0.3, 0.5, 0.7):
                f = fp_at_rec(ev, rec_t)
                oper.append({
                    "method": name, "mode": mode,
                    "fp_per_frame": None if f is None else round(f, 4),
                    "novel_recall": rec_t, "known_recall": None,
                    "valid_recall": None,
                })
    with open(OUT / "full_stream_tco_gate_curve.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
        w.writeheader()
        w.writerows(curves)
    with open(OUT / "full_stream_tco_gate_operating.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(oper[0].keys()))
        w.writeheader()
        w.writerows(oper)
    print("\n=== novel recall @ FP/frame ===")
    for r in oper:
        if r["known_recall"] is not None:
            print(f"{r['method']:14s} {r['mode']:8s} "
                  f"FP={r['fp_per_frame']:5.2f} "
                  f"novel_rec={r['novel_recall']} "
                  f"known_rec={r['known_recall']} "
                  f"valid_rec={r['valid_recall']}")
    print("\n=== FP/frame @ novel recall ===")
    for r in oper:
        if r["known_recall"] is None:
            print(f"{r['method']:14s} {r['mode']:8s} "
                  f"rec={r['novel_recall']} FP/frame={r['fp_per_frame']}")


if __name__ == "__main__":
    main()
