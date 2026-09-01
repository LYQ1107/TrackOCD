#!/usr/bin/env python3
"""Audit COVTrack trajectory-conditioned objectness gate (B-route).

Metrics follow the full-stream protocol used for D0/OVTR/COVTrack:
  - dev: 732 frames, heldout: 887 frames;
  - proposals are scored and thresholded globally; FP/frame = cumulative FP
    over split frames;
  - LR models are trained on dev only (5-fold OOF for dev curves,
    dev-fit transfer to heldout); nothing is tuned on heldout.

Feature sets:
  B1 baseline_score: current raw score
  B2 age_confirmation: prior hits / age / recency confirmation
  B3 causal_only: B2 + causal motion/appearance-score history (no category
     semantics)
  B4 causal_static: B3 + current-frame static/visual features
  B5 causal_static_semantic: B4 + current and prior semantic confidence
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4p" / "covtrack_tco"
N_FRAMES = {"dev": 732, "heldout": 887}

B1 = ["score"]
B2 = ["prior_hits", "prior_age", "recent_hit_ratio", "consecutive_hits",
      "recent_miss"]
B3 = B2 + ["prior_score_mean", "prior_score_std",
           "prior_area_mean", "prior_area_std",
           "disp_mean", "disp_std",
           "scale_delta_mean", "scale_delta_std",
           "app_sim_mean", "app_sim_std"]
B4 = B3 + ["bbox_area_log", "bbox_aspect_log", "dino_norm"]
B5 = B4 + ["best_known", "known_margin",
           "prior_best_known_mean", "prior_best_known_std",
           "prior_margin_mean", "prior_margin_std"]

FEATURE_SETS = {
    "baseline_score": B1,
    "age_confirmation": B2,
    "causal_only": B3,
    "causal_static": B4,
    "causal_static_semantic": B5,
}

AGE_BUCKETS = [
    ("age0", 0, 0),
    ("age1", 1, 1),
    ("age2", 2, 2),
    ("age3_4", 3, 4),
    ("age5plus", 5, 10 ** 9),
]


def load_rows(mode):
    rows = list(csv.DictReader(open(OUT / f"causal_features_{mode}.csv")))
    for r in rows:
        for k in ("video_id", "frame_id", "image_id", "proposal_local_id",
                  "track_id"):
            r[k] = int(r[k])
        r["gt_role"] = r["gt_role"].strip().lower()
        for k in ("score", "bbox_area_log", "bbox_aspect_log", "dino_norm",
                  "best_known", "known_margin", "prior_age", "prior_hits",
                  "recent_hit_ratio", "consecutive_hits", "recent_miss",
                  "prior_score_mean", "prior_score_std", "prior_area_mean",
                  "prior_area_std", "disp_mean", "disp_std",
                  "scale_delta_mean", "scale_delta_std",
                  "app_sim_mean", "app_sim_std",
                  "prior_best_known_mean", "prior_best_known_std",
                  "prior_margin_mean", "prior_margin_std"):
            r[k] = float(r[k])
    return rows


def auroc(y, s):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    pos = s[y]
    neg = s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    u = ranks[y].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def auprc(y, s):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(y.sum()), 1)
    ap = 0.0
    prev = 0.0
    for r, p in zip(rec, prec):
        ap += (r - prev) * p
        prev = r
    return float(ap)


def matrix(rows, cols, persistent_fp_only=False, include_known=False):
    X, y, idx = [], [], []
    for i, r in enumerate(rows):
        role = r["gt_role"]
        if role == "known" and include_known:
            pass
        elif role == "novel":
            pass
        elif role == "fp" and (not persistent_fp_only or r["prior_hits"] >= 2):
            pass
        else:
            continue
        X.append([r[c] for c in cols])
        y.append(1 if role == "novel" else 0)
        idx.append(i)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        if np.isnan(med):
            med = 0.0
        col[np.isnan(col)] = med
    return X, y, idx


def train_oof(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = np.zeros(len(X), dtype=np.float64)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.1, max_iter=3000).fit(
            sc.transform(X[tr]), y[tr])
        scores[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return scores


def train_transfer_scores(Xd, yd, Xh):
    sc = StandardScaler().fit(Xd)
    clf = LogisticRegression(C=0.1, max_iter=3000).fit(
        sc.transform(Xd), yd)
    return clf.predict_proba(sc.transform(Xh))[:, 1]


def full_curve(rows, scores, mode):
    n_frames = N_FRAMES[mode]
    order = np.argsort(-np.asarray(scores, dtype=np.float64),
                       kind="mergesort")
    roles = np.asarray([r["gt_role"] for r in rows])
    prior_hits = np.asarray([r["prior_hits"] for r in rows])
    s = np.asarray(scores)[order]
    r = roles[order]
    ph = prior_hits[order]
    cum_novel = np.cumsum(r == "novel")
    cum_known = np.cumsum(r == "known")
    cum_fp = np.cumsum(r == "fp")
    cum_pfp = np.cumsum((r == "fp") & (ph >= 2))
    total_novel = int((r == "novel").sum())
    total_fp = int((r == "fp").sum())
    total_pfp = int(((r == "fp") & (ph >= 2)).sum())
    novel_recall = cum_novel / max(total_novel, 1)
    known_recall = cum_known / max(int((r == "known").sum()), 1)
    fp_per_frame = cum_fp / n_frames

    def recall_at_fp(target):
        idx = np.where(fp_per_frame <= target)[0]
        return float(novel_recall[idx[-1]]) if len(idx) else 0.0

    def fp_at_recall(target):
        idx = np.where(novel_recall >= target)[0]
        return float(fp_per_frame[idx[0]]) if len(idx) else None

    def cutoff_at_recall(target):
        idx = np.where(novel_recall >= target)[0]
        return int(idx[0]) if len(idx) else None

    def cutoff_at_fp(target):
        idx = np.where(fp_per_frame <= target)[0]
        return int(idx[-1]) if len(idx) else None

    def reject_at(cut):
        if cut is None:
            return (None, None)
        kept_fp = int(cum_fp[cut])
        kept_pfp = int(cum_pfp[cut])
        all_rej = 1.0 - kept_fp / total_fp if total_fp else None
        pfp_rej = 1.0 - kept_pfp / total_pfp if total_pfp else None
        return (all_rej, pfp_rej)

    age_totals = {}
    age_cum = {}
    for name, lo, hi in AGE_BUCKETS:
        age_totals[name] = int(((r == "novel") & (ph >= lo) &
                                (ph <= hi)).sum())
        age_cum[name] = np.cumsum((r == "novel") & (ph >= lo) & (ph <= hi))

    m = {
        "mode": mode,
        "total_novel": total_novel,
        "total_fp": total_fp,
        "total_persistent_fp": total_pfp,
        "recall_at_fp_0_3": recall_at_fp(0.3),
        "recall_at_fp_1": recall_at_fp(1.0),
        "recall_at_fp_3": recall_at_fp(3.0),
        "known_recall_at_fp_1": float(
            known_recall[cutoff_at_fp(1.0)]) if cutoff_at_fp(1.0) is not None
        else 0.0,
        "fp_per_frame_at_recall_0_3": fp_at_recall(0.3),
        "fp_per_frame_at_recall_0_5": fp_at_recall(0.5),
        "fp_per_frame_at_recall_0_7": fp_at_recall(0.7),
    }
    for label, cut_fn, cut_name in (
            ("fp1", cutoff_at_fp, 1.0), ("fp3", cutoff_at_fp, 3.0),
            ("r03", cutoff_at_recall, 0.3),
            ("r05", cutoff_at_recall, 0.5)):
        cut = cut_fn(cut_name)
        all_rej, pfp_rej = reject_at(cut)
        m[f"reject_all_fp_at_{label}"] = all_rej
        m[f"reject_persistent_fp_at_{label}"] = pfp_rej
        for name, lo, hi in AGE_BUCKETS:
            if cut is None:
                m[f"early_{name}_recall_at_{label}"] = None
            elif age_totals[name]:
                m[f"early_{name}_recall_at_{label}"] = float(
                    age_cum[name][cut] / age_totals[name])
            else:
                m[f"early_{name}_recall_at_{label}"] = None

    curve_rows = []
    for i in range(len(order)):
        curve_rows.append({
            "rank": i,
            "score": float(s[i]),
            "novel_recall": float(novel_recall[i]),
            "known_recall": float(known_recall[i]),
            "fp_per_frame": float(fp_per_frame[i]),
            "cum_fp": int(cum_fp[i]),
        })
    return m, curve_rows


def main():
    dev = load_rows("dev")
    ho = load_rows("heldout")
    summary = []
    pareto = {"dev": [], "heldout": []}
    early_rows = []

    for name, cols in FEATURE_SETS.items():
        Xd, yd, dev_tr_idx = matrix(dev, cols)
        Xh, yh, _ = matrix(ho, cols)
        oof = train_oof(Xd, yd)
        # Full-fit dev model scores every row (known rows are not in the
        # training objective but are still part of the full-stream curve).
        full_sc = StandardScaler().fit(Xd)
        full_clf = LogisticRegression(C=0.1, max_iter=3000).fit(
            full_sc.transform(Xd), yd)
        Xd_all, _, _ = matrix(dev, cols, include_known=True)
        Xh_all, _, _ = matrix(ho, cols, include_known=True)
        dev_scores = np.full(len(dev), np.nan, dtype=np.float64)
        dev_scores[dev_tr_idx] = oof
        known_mask = np.asarray([r["gt_role"] == "known" for r in dev])
        full_dev = full_clf.predict_proba(full_sc.transform(Xd_all))[:, 1]
        dev_scores[known_mask] = full_dev[known_mask]
        ho_scores = full_clf.predict_proba(full_sc.transform(Xh_all))[:, 1]

        Xd_p, yd_p, _ = matrix(dev, cols, persistent_fp_only=True)
        Xh_p, yh_p, _ = matrix(ho, cols, persistent_fp_only=True)
        oof_p = train_oof(Xd_p, yd_p) if len(np.unique(yd_p)) == 2 else None
        tr_p = (train_transfer_scores(Xd_p, yd_p, Xh_p)
                if len(np.unique(yd_p)) == 2 and len(np.unique(yh_p)) == 2
                else None)

        for mode, rows, scores in (("dev", dev, dev_scores),
                                   ("heldout", ho, ho_scores)):
            m, curves = full_curve(rows, scores, mode)
            Xm, ym, ym_idx = matrix(rows, cols)
            m["auroc"] = auroc(ym, scores[ym_idx])
            m["auprc"] = auprc(ym, scores[ym_idx])
            Xp, yp, yp_idx = matrix(rows, cols, persistent_fp_only=True)
            if len(np.unique(yp)) == 2:
                scores_p = oof_p if mode == "dev" and oof_p is not None \
                    else tr_p
                if scores_p is not None:
                    m["persistent_auroc"] = auroc(yp, scores_p)
                    m["persistent_auprc"] = auprc(yp, scores_p)
                else:
                    m["persistent_auroc"] = None
                    m["persistent_auprc"] = None
            else:
                m["persistent_auroc"] = None
                m["persistent_auprc"] = None
            m["method"] = name
            summary.append(m)
            for c in curves:
                c["method"] = name
                c["mode"] = mode
                pareto[mode].append(c)

        for mode, rows, scores in (("dev", dev, dev_scores),
                                   ("heldout", ho, ho_scores)):
            m, _ = full_curve(rows, scores, mode)
            for bname, _, _ in AGE_BUCKETS:
                early_rows.append({
                    "method": name,
                    "mode": mode,
                    "bucket": bname,
                    "recall_at_fp1": m[f"early_{bname}_recall_at_fp1"],
                    "recall_at_fp3": m[f"early_{bname}_recall_at_fp3"],
                    "recall_at_novel_recall_0_3": m[
                        f"early_{bname}_recall_at_r03"],
                })

    fieldnames = [
        "method", "mode", "auroc", "auprc", "persistent_auroc",
        "persistent_auprc", "total_novel", "total_fp", "total_persistent_fp",
        "recall_at_fp_0_3", "recall_at_fp_1", "recall_at_fp_3",
        "known_recall_at_fp_1", "fp_per_frame_at_recall_0_3",
        "fp_per_frame_at_recall_0_5", "fp_per_frame_at_recall_0_7",
        "reject_all_fp_at_fp1", "reject_persistent_fp_at_fp1",
        "reject_all_fp_at_r03", "reject_persistent_fp_at_r03",
        "reject_all_fp_at_r05", "reject_persistent_fp_at_r05",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "covtrack_tco_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary:
            w.writerow({k: r.get(k) for k in fieldnames})
    for name in FEATURE_SETS:
        rows = [r for r in summary if r["method"] == name]
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})
    for mode in ("dev", "heldout"):
        with open(OUT / f"pareto_{mode}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "method", "mode", "rank", "score", "novel_recall",
                "known_recall",
                "fp_per_frame", "cum_fp"])
            w.writeheader()
            w.writerows(pareto[mode])
    with open(OUT / "early_novel_by_age.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(early_rows[0].keys()))
        w.writeheader()
        w.writerows(early_rows)
    with open(OUT / "covtrack_tco_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("COVTRACK_TCO_AUDIT_DONE")
    for r in summary:
        print(f"{r['method']:24s} {r['mode']:8s} "
              f"AUROC={r['auroc']:.4f} AUPRC={r['auprc']:.4f} "
              f"rec@1FP={r['recall_at_fp_1']:.4f} "
              f"FP@r0.3={r['fp_per_frame_at_recall_0_3']} "
              f"rejectPFP@r0.3={r['reject_persistent_fp_at_r03']}")


if __name__ == "__main__":
    main()
