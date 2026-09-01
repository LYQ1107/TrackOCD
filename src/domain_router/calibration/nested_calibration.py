from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.domain_router.features.router_features import compute_router_features
from src.domain_router.models.routers import FEATURE_SETS, make_router


def hmean(kr, nr):
    return 2.0 * kr * nr / (kr + nr) if kr + nr > 0 else 0.0


def build_feature_matrix(ids, feats, protos, knn_index, frame_feats, meta, names):
    rows = []
    for s in ids:
        f = compute_router_features(
            feats[s], protos, knn_index,
            frame_feats.get(s) if frame_feats else None,
            meta.get(s) if meta else None,
        )
        rows.append([f[n] for n in names])
    return np.asarray(rows, dtype=np.float64)


def fit_router_on_fold(name, feats, protos, fold, knn_index, frame_feats, meta,
                       known_recall_floor=None):
    """Fit candidate on source fold ids; select threshold on target ids.
    protos must be built from source proxy-known classes only."""
    src_pos = [s for s in fold["source_positive_ids"] if s in feats]
    src_neg = [s for s in fold["source_negative_ids"] if s in feats]
    tgt_pos = [s for s in fold["target_positive_ids"] if s in feats]
    tgt_neg = [s for s in fold["target_negative_ids"] if s in feats]
    if name == "R0":
        router = make_router("R0", protos, 0.45)
        scores_src = None
    elif name == "R1":
        router = make_router("R1", protos, 0.45, 0.0)
        scores_src = None
    else:
        tr_ids = src_pos + src_neg
        y_tr = np.array([1] * len(src_pos) + [0] * len(src_neg))
        X_tr = build_feature_matrix(tr_ids, feats, protos, knn_index,
                                    frame_feats, meta, FEATURE_SETS[name])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(X_tr, y_tr)
        router = make_router(name, protos, 0.5,
                             coef=lr.coef_[0], intercept=lr.intercept_[0])
    # target scores
    t_ids = tgt_pos + tgt_neg
    y_t = np.array([1] * len(tgt_pos) + [0] * len(tgt_neg))
    scores = []
    for s in t_ids:
        f = compute_router_features(feats[s], protos, knn_index,
                                    frame_feats.get(s) if frame_feats else None,
                                    meta.get(s) if meta else None)
        if name == "R0":
            scores.append(f["s1"])
        elif name == "R1":
            scores.append(f["s1"] if f["margin"] >= 0.0 else -1.0)
        else:
            scores.append(router.score(f))
    scores = np.asarray(scores)
    # R0 reference known recall on the same target fold (s1 >= 0.45)
    r0_kr = 0.0
    if len(tgt_pos):
        r0_kr = 0.0
        cnt = 0
        for s in tgt_pos:
            f = compute_router_features(feats[s], protos, knn_index,
                                        frame_feats.get(s) if frame_feats else None,
                                        meta.get(s) if meta else None)
            cnt += 1 if f["s1"] >= 0.45 else 0
        r0_kr = cnt / len(tgt_pos)
    if known_recall_floor is None:
        known_recall_floor = max(0.0, r0_kr - 0.03)
    best = None
    best_kr_fallback = None
    for thr in np.arange(0.20, 0.96, 0.01):
        pred = scores >= thr
        kr = pred[y_t == 1].mean() if (y_t == 1).any() else 0.0
        nr = (~pred[y_t == 0]).mean() if (y_t == 0).any() else 0.0
        hm = hmean(kr, nr)
        if best_kr_fallback is None or kr > best_kr_fallback[1]:
            best_kr_fallback = (thr, kr, nr, hm)
        if known_recall_floor is not None and kr < known_recall_floor:
            continue
        if best is None or hm > best[0]:
            best = (hm, kr, nr, float(thr))
    if best is None and best_kr_fallback is not None:
        thr, kr, nr, hm = best_kr_fallback
        best = (hm, kr, nr, float(thr))
    return router, {
        "hmean": best[0] if best else 0.0,
        "known_recall": best[1] if best else 0.0,
        "novel_recall": best[2] if best else 0.0,
        "threshold": best[3] if best else 0.45,
    }


def nested_select_router(folds, feats, protos_by_fold, knn_by_fold,
                         frame_feats, meta):
    rows = []
    for name in ("R0", "R1", "R2", "R3", "R4"):
        for i, fold in enumerate(folds):
            router, m = fit_router_on_fold(
                name, feats, protos_by_fold[i], fold, knn_by_fold[i],
                frame_feats, meta,
            )
            rows.append({
                "router": name, "target_domain": fold["target_domain"],
                "fold_index": i, **m,
            })
    return rows
