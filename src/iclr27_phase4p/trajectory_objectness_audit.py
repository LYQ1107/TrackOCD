"""Phase 4P Q1 audit: frame-online, detection-level separation of
TRUE_NOVEL from FP (incl. PERSISTENT_FP) using only:
  - current-frame visual/semantic evidence (frame t)
  - strictly historical prefix of the same physical track (frames <= t-1)

No future access: every trajectory feature for a row is computed from
rows with the same physical_track_id and frame_id < current frame.
"""
from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4p"


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def auroc(y, s):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    pos = s[y]
    neg = s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    u = ranks[y].sum() - len(pos) * (len(pos) + 1) / 2
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


STATIC_FEATS = [
    "score", "mask_frac", "app_norm", "dino_norm", "z_norm",
    "known_margin", "p_known", "best_known", "best_novel",
]
CAUSAL_FEATS = [
    "prior_hits", "prior_score_mean", "prior_score_std",
    "prior_score_trend", "prior_pk_mean", "prior_pk_std",
    "prior_known_margin_mean", "prior_mask_mean", "prior_app_mean",
    "prior_max_gap",
]


def add_causal_features(rows):
    """Rows must be sorted by (video_id, frame_id, det_local_id)."""
    hist = defaultdict(list)  # (video_id, track_id) -> list of prior rows
    out = []
    for r in rows:
        tid = r.get("track_id", "")
        key = (r["video_id"], tid)
        prior = hist.get(key, [])
        f = {"prior_hits": len(prior)}
        if prior:
            scores = [num(p["score"]) for p in prior]
            pks = [num(p["p_known"]) for p in prior]
            kms = [num(p["known_margin"]) for p in prior]
            masks = [num(p["mask_frac"]) for p in prior]
            apps = [num(p["app_norm"]) for p in prior]
            f["prior_score_mean"] = float(np.nanmean(scores))
            f["prior_score_std"] = float(np.nanstd(scores))
            f["prior_score_trend"] = float(scores[-1] - scores[0])
            f["prior_pk_mean"] = float(np.nanmean(pks))
            f["prior_pk_std"] = float(np.nanstd(pks))
            f["prior_known_margin_mean"] = float(np.nanmean(kms))
            f["prior_mask_mean"] = float(np.nanmean(masks))
            f["prior_app_mean"] = float(np.nanmean(apps))
            fr = [int(p["frame_id"]) for p in prior]
            f["prior_max_gap"] = float(max(
                fr[i] - fr[i - 1] - 1 for i in range(1, len(fr)))) \
                if len(fr) >= 2 else 0.0
        else:
            for k in ("prior_score_mean", "prior_score_std",
                      "prior_score_trend", "prior_pk_mean", "prior_pk_std",
                      "prior_known_margin_mean", "prior_mask_mean",
                      "prior_app_mean", "prior_max_gap"):
                f[k] = 0.0
        for k, v in f.items():
            r[k] = v
        if tid not in ("", None):
            hist[key].append(r)
        out.append(r)
    return out


def role_of(r):
    g = r.get("gt_role", "")
    if g == "known":
        return "KNOWN"
    if g == "novel":
        return "NOVEL"
    return "FP"


def load_pop(path, mode):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["mode"] = mode
        for k in STATIC_FEATS:
            r[k] = num(r.get(k))
        r["frame_id"] = int(r["frame_id"])
        r["video_id"] = int(r["video_id"])
        r["det_local_id"] = int(r["det_local_id"])
    rows.sort(key=lambda r: (r["video_id"], r["frame_id"],
                             r["det_local_id"]))
    return add_causal_features(rows)


def mat(rows, cols, fp_mode):
    X, y, meta = [], [], []
    for r in rows:
        if role_of(r) == "NOVEL":
            ok = True
        elif role_of(r) == "FP":
            ok = fp_mode == "all_FP" or r["prior_hits"] >= 2
        else:
            ok = False
        if ok:
            X.append([r[c] for c in cols])
            y.append(1 if role_of(r) == "NOVEL" else 0)
            meta.append((r["mode"], r["video_id"], r["frame_id"],
                         r["det_local_id"]))
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(med))[0]
    med[inds] = 0.0
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = med[j]
    return X, y, meta


def cross_val_auroc(X, y, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(0)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    aurocs, auprcs = [], []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]),
                                                    y[tr])
        s = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aurocs.append(auroc(y[te], s) or 0.0)
        auprcs.append(auprc(y[te], s))
    return float(np.mean(aurocs)), float(np.mean(auprcs))


def train_eval_transfer(Xd, yd, Xh, yh):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xd)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xd), yd)
    s = clf.predict_proba(sc.transform(Xh))[:, 1]
    return auroc(yh, s), auprc(yh, s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = load_pop(
        ROOT / "outputs" / "iclr27_phase4p" / "audit" /
        "detection_population_dev_fixed.csv", "dev")
    ho = load_pop(
        ROOT / "outputs" / "iclr27_phase4p" / "audit" /
        "detection_population_heldout_corrected_fixed.csv", "heldout")

    # Detection-level audit CSV.
    with open(OUT / "dev" / "trajectory_objectness_audit.csv",
              "w", newline="") as f:
        cols = ["mode", "video_id", "frame_id", "det_local_id", "score",
                "mask_frac", "app_norm", "dino_norm", "z_norm",
                "known_margin", "p_known", "best_known", "best_novel",
                "track_id", "track_age", "prior_hits", "prior_score_mean",
                "prior_score_std", "prior_score_trend", "prior_pk_mean",
                "prior_pk_std", "prior_known_margin_mean",
                "prior_mask_mean", "prior_app_mean", "prior_max_gap",
                "gt_role", "gt_category", "gt_track_id"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in dev:
            w.writerow({k: r.get(k, "") for k in cols})
    with open(OUT / "heldout" / "trajectory_objectness_audit.csv",
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ho:
            w.writerow({k: r.get(k, "") for k in cols})

    # Per-frame novel coverage by prior_hits bucket (diagnostic).
    age_rows = []
    for label, rows in (("dev", dev), ("heldout", ho)):
        for bucket, lo, hi in (("age0", 0, 0), ("age1", 1, 1),
                               ("age2", 2, 2), ("age3_4", 3, 4),
                               ("age5+", 5, 10 ** 9)):
            nov = [r for r in rows if role_of(r) == "NOVEL" and
                   lo <= r["prior_hits"] <= hi]
            fp = [r for r in rows if role_of(r) == "FP" and
                  lo <= r["prior_hits"] <= hi]
            age_rows.append({
                "mode": label, "bucket": bucket,
                "novel_rows": len(nov), "fp_rows": len(fp),
            })
    with open(OUT / "dev" / "novel_age_coverage.csv",
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(age_rows[0].keys()))
        w.writeheader()
        w.writerows(age_rows)

    # Feature-set comparison: dev 5-fold CV + dev->heldout transfer.
    summary = []
    sets = {
        "static": STATIC_FEATS,
        "causal_trajectory": CAUSAL_FEATS,
        "all": STATIC_FEATS + CAUSAL_FEATS,
    }
    for fp_mode in ("all_FP", "persistent_FP"):
        for name, cols in sets.items():
            Xd, yd, _ = mat(dev, cols, fp_mode)
            Xh, yh, _ = mat(ho, cols, fp_mode)
            if len(set(yd.tolist())) < 2 or len(set(yh.tolist())) < 2:
                continue
            a_d, p_d = cross_val_auroc(Xd, yd)
            a_h, p_h = train_eval_transfer(Xd, yd, Xh, yh)
            summary.append({
                "features": name, "fp_mode": fp_mode,
                "dev_n_novel": int((yd == 1).sum()),
                "dev_n_fp": int((yd == 0).sum()),
                "ho_n_novel": int((yh == 1).sum()),
                "ho_n_fp": int((yh == 0).sum()),
                "dev_CV_AUROC": round(a_d, 4),
                "dev_CV_AUPRC": round(p_d, 4),
                "ho_transfer_AUROC": round(a_h, 4),
                "ho_transfer_AUPRC": round(p_h, 4),
            })
    with open(OUT / "dev" / "trajectory_signal_predictability.csv",
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # Frame-online track-confirmation heuristic baseline.
    heur = []
    for label, rows in (("dev", dev), ("heldout", ho)):
        for k in (0, 1, 2, 4):
            for s_th in (0.5, 0.6, 0.7):
                keep = [r for r in rows
                        if r["prior_hits"] >= k and r["score"] >= s_th]
                n_novel = sum(1 for r in keep if role_of(r) == "NOVEL")
                n_fp = sum(1 for r in keep if role_of(r) == "FP")
                n_known = sum(1 for r in keep if role_of(r) == "KNOWN")
                total_novel = sum(1 for r in rows
                                  if role_of(r) == "NOVEL")
                heur.append({
                    "mode": label,
                    "rule": f"prior_hits>={k}&score>={s_th}",
                    "kept_rows": len(keep), "novel": n_novel,
                    "fp": n_fp, "known": n_known,
                    "novel_precision": round(
                        n_novel / max(len(keep), 1), 4),
                    "novel_recall": round(
                        n_novel / max(total_novel, 1), 4),
                })
    with open(OUT / "dev" / "track_confirmation_baseline.csv",
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(heur[0].keys()))
        w.writeheader()
        w.writerows(heur)

    print("TRAJECTORY_AUDIT_DONE")
    for r in summary:
        print(r["features"], r["fp_mode"],
              "devCV", r["dev_CV_AUROC"], r["dev_CV_AUPRC"],
              "hoTransfer", r["ho_transfer_AUROC"],
              r["ho_transfer_AUPRC"])


if __name__ == "__main__":
    main()
