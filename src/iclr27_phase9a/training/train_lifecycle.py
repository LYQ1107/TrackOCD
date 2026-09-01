"""Fit the small causal lifecycle heads for Phase 9A.

This is a legal meta-learning preparation step, not a large backbone
training run.  Foundation h-vectors are frozen.  The action head learns
known / novel / false-birth evidence; the maturity and pair heads learn from
causal trajectory summaries and category-held-out novel episodes.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

from src.iclr27_phase9a.lifecycle import unit
from src.iclr27_phase7a.training.train_reliability_head import load_tse

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def load_episode(name: str):
    p = ROOT / "outputs/iclr27_phase7c/assets" / f"{name}.npz"
    return {k: np.asarray(v) for k, v in np.load(p).items()}


def load_foundation(n: int) -> np.ndarray:
    h = np.asarray(np.load(ROOT / "outputs/iclr27_phase7c/assets/h_all.npz")["h"],
                   dtype=np.float32)
    if len(h) < n:
        raise RuntimeError(f"h_all has {len(h)} rows, need {n}")
    return h[:n].copy()


def chrono_order(ep: dict) -> np.ndarray:
    return np.lexsort((ep["proposal_local_ids"], ep["frame_ids"],
                       ep["video_ids"]))


def causal_tracks(ep: dict, h: np.ndarray):
    """Return h/age/uncertainty/consistency computed prefix-causally."""
    n, d = h.shape
    out_h = np.zeros_like(h)
    age = np.zeros(n, dtype=np.int32)
    unc = np.zeros(n, dtype=np.float32)
    con = np.zeros(n, dtype=np.float32)
    state = {}
    for i in chrono_order(ep):
        key = (int(ep["video_ids"][i]), int(ep["track_ids"][i]))
        z = unit(h[i])
        old = state.get(key)
        if old is None:
            mean, count, variance = z.copy(), 1, 0.0
            consistency = 1.0
        else:
            mean, count, variance = old
            consistency = float(np.dot(mean, z))
            delta = z - mean
            variance = (variance * count + float(np.mean(delta * delta))) / (count + 1)
            mean = unit((mean * count + z) / float(count + 1))
            count += 1
        state[key] = (mean, count, variance)
        out_h[i] = mean
        age[i] = count
        unc[i] = variance
        con[i] = consistency
    return out_h, age, unc, con


def known_prototypes(ep: dict, h: np.ndarray, visible: list[int]):
    protos = []
    for c in visible:
        mask = (ep["gt_category_id"] == int(c)) & (ep["gt_role"] == 1)
        if not np.any(mask):
            raise RuntimeError(f"missing legal known rows for category {c}")
        protos.append(unit(h[mask].mean(axis=0)))
    return np.asarray(protos, dtype=np.float32)


def foundation_known_prototypes():
    """Frozen 48-class foundation anchors, legal known-state initialization."""
    _, anchors, known_ids = load_tse(torch.device("cpu"))
    anchors = np.asarray(anchors, dtype=np.float32)
    anchors /= np.maximum(np.linalg.norm(anchors, axis=1, keepdims=True), 1e-8)
    return anchors, [int(x) for x in np.asarray(known_ids).tolist()]


def action_matrix(ep: dict, h: np.ndarray, age, unc, con, protos):
    sims = h @ protos.T
    order = np.argsort(-sims, axis=1)
    best = sims[np.arange(len(h)), order[:, 0]]
    second = sims[np.arange(len(h)), order[:, 1]]
    return np.c_[best, second, best - second,
                  np.minimum(age, 50).astype(np.float32) / 50.0,
                  unc.astype(np.float32), con.astype(np.float32)].astype(np.float32)


def maturity_matrix(ep: dict, h: np.ndarray, age, unc, con,
                    protos: np.ndarray) -> np.ndarray:
    """Build state features exactly as the online candidate state does.

    A candidate is owned by one physical track until it becomes reusable, so
    its evidence count is the causal track age and its support-track count is
    zero at birth and one after the first update.  Crucially, knownness is
    frozen at the candidate's birth; using the current-frame knownness here
    would train a feature different from :func:`maturity_features` at replay.
    """
    birth_known = np.zeros(len(h), dtype=np.float32)
    seen = {}
    for i in chrono_order(ep):
        key = (int(ep["video_ids"][i]), int(ep["track_ids"][i]))
        if key not in seen:
            seen[key] = float(np.max(h[i] @ protos.T))
        birth_known[i] = seen[key]
    support = (np.asarray(age, dtype=np.float32) >= 2.0).astype(np.float32)
    return np.c_[
        np.log1p(np.asarray(age, dtype=np.float32)),
        np.log1p(np.asarray(age, dtype=np.float32)),
        np.asarray(unc, dtype=np.float32),
        np.asarray(con, dtype=np.float32),
        1.0 - birth_known,
        np.minimum(support, 20.0) / 20.0,
    ].astype(np.float32)


def fit_action_legacy(X: np.ndarray, y: np.ndarray, novel_weight: float):
    """Retained only for historical ablation reproduction."""
    clf = LogisticRegression(
        C=1.0, max_iter=500, multi_class="multinomial",
        class_weight={0: 1.0, 1: float(novel_weight), 2: 0.2},
        random_state=2717)
    clf.fit(X, y)
    return clf.coef_.astype(np.float32), clf.intercept_.astype(np.float32)


def fit_action_binary(X: np.ndarray, known: np.ndarray,
                      known_weight: float):
    """Fit a binary semantic knownness evidence head.

    The online lifecycle has only two evidence outcomes: known or
    open-world.  Novel births and detector false births are both public
    ``NEW_NOVEL`` candidates; the maturity head, trained with explicit false
    births, decides when either candidate may become reusable.  This avoids
    reintroducing the rejected three-way identity head.
    """
    clf = LogisticRegression(
        C=1.0, max_iter=500, class_weight={0: 1.0, 1: float(known_weight)},
        random_state=2717)
    clf.fit(X, known.astype(np.int64))
    if list(clf.classes_) != [0, 1]:
        raise RuntimeError(f"binary knownness classes={clf.classes_}")
    # LifecycleHeads uses row 0 for KNOWN and row 1 for OPEN.  The positive
    # sklearn log-odds are KNOWN-vs-OPEN; comparing them with the zero OPEN
    # logit is a learned evidence decision, not a fixed prototype gate.
    w = np.vstack([clf.coef_[0], np.zeros(X.shape[1], dtype=np.float32)])
    b = np.asarray([clf.intercept_[0], 0.0], dtype=np.float32)
    return w.astype(np.float32), b


def fit_maturity(ep: dict, h: np.ndarray, age, unc, con, protos,
                  visible: list[int], include_fp_negatives: bool = True):
    novel = ep["row_split"] == 1
    cats = np.unique(ep["gt_category_id"][novel]).astype(int)
    cat_proto = {}
    quality = {}
    sims_known = h @ protos.T
    # Keep this matrix aligned with the online state lifecycle.  The labels
    # below may use current-frame semantic quality, but the input features
    # must be prefix-causal birth/trajectory summaries.
    Xm_all = maturity_matrix(ep, h, age, unc, con, protos)
    for c in cats:
        m = novel & (ep["gt_category_id"] == c)
        p = unit(h[m].mean(axis=0))
        cat_proto[c] = p
        quality[c] = h[m] @ p - np.max(sims_known[m], axis=1)
    # The supervision is a semantic evidence event, not a frame counter:
    # each category gets its own quality quantile, and dispersion is also
    # required.  The head then learns how evidence count/uncertainty predicts
    # this event on unseen categories.
    X, y = [], []
    positive_ages = []
    for c in cats:
        m = novel & (ep["gt_category_id"] == c)
        q = quality[c]
        q_cut = float(np.quantile(q, 0.55))
        u_cut = float(np.quantile(unc[m], 0.65))
        labels = (q >= q_cut) & (unc[m] <= u_cut) & (age[m] >= 2)
        X.append(Xm_all[m])
        y.append(labels.astype(np.int64))
        positive_ages.extend(np.asarray(age[m])[labels].astype(int).tolist())
    # Explicit false-birth negatives: a state that is born from a detector
    # FP/noisy fragment must not become reusable merely because it is long.
    fp = ep["gt_role"] == 0 if include_fp_negatives else np.zeros(
        len(ep["gt_role"]), dtype=bool)
    if np.any(fp):
        # Match the negative pool to the number of positive evidence events;
        # otherwise the 30k FP rows make duration alone dominate the learned
        # maturity score.  Match the causal age distribution as well, so the
        # head cannot learn the invalid rule "longer means reusable".
        n_pos = max(len(positive_ages), 1)
        fp_idx = np.flatnonzero(fp)
        n_neg = min(len(fp_idx), max(4 * n_pos, 256))
        rng = np.random.default_rng(2717)
        if positive_ages:
            target_hist = np.bincount(
                np.minimum(np.asarray(positive_ages, dtype=np.int64), 20),
                minlength=21).astype(np.float64)
            fp_age = np.minimum(np.asarray(age[fp_idx], dtype=np.int64), 20)
            pool_hist = np.bincount(fp_age, minlength=21).astype(np.float64)
            weights = target_hist[fp_age] / np.maximum(pool_hist[fp_age], 1.0)
            weights /= max(float(weights.sum()), 1e-12)
            fp_idx = rng.choice(fp_idx, size=n_neg, replace=False, p=weights)
        else:
            fp_idx = rng.choice(fp_idx, size=n_neg, replace=False)
        X.append(Xm_all[fp_idx])
        y.append(np.zeros(int(n_neg), dtype=np.int64))
    X = np.concatenate(X, axis=0).astype(np.float32)
    y = np.concatenate(y, axis=0)
    if len(np.unique(y)) < 2:
        # This should not occur for the legal hard episodes, but keeps a
        # deterministic identity fallback for a tiny smoke subset.
        return np.zeros(X.shape[1], np.float32), np.float32(-1.0), X, y
    clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced",
                             random_state=2717)
    clf.fit(X, y)
    return clf.coef_[0].astype(np.float32), np.float32(clf.intercept_[0]), X, y


def fit_reuse(ep: dict, h: np.ndarray, age, unc, protos):
    """Construct causal same-category / different-category pair examples."""
    X, y = [], []
    cat_mean = {}
    cat_count = defaultdict(int)
    cat_unc = defaultdict(float)
    seen_cats = []
    for i in chrono_order(ep):
        if int(ep["row_split"][i]) != 1:
            continue
        c = int(ep["gt_category_id"][i])
        sims = h[i] @ protos.T
        so = np.sort(sims)[::-1]
        known_best, known_margin = float(so[0]), float(so[0] - so[1])
        # A positive query can only reuse a state born from an earlier
        # occurrence of the same category.
        if c in cat_mean:
            proto = cat_mean[c]
            X.append([float(h[i] @ proto),
                      0.5,
                      np.log1p(cat_count[c]), cat_unc[c],
                      min(int(age[i]), 50) / 50.0, float(unc[i]),
                      known_best, known_margin])
            y.append(1)
            # One hard negative from an already-seen different category.
            if seen_cats:
                other = seen_cats[(cat_count[c] + i) % len(seen_cats)]
                op = cat_mean[other]
                X.append([float(h[i] @ op), 0.5,
                          np.log1p(cat_count[other]), cat_unc[other],
                          min(int(age[i]), 50) / 50.0, float(unc[i]),
                          known_best, known_margin])
                y.append(0)
        # Update only after constructing the query: no future evidence leaks
        # into the pair feature.
        if c not in cat_mean:
            seen_cats.append(c)
            cat_mean[c] = h[i].copy()
            cat_count[c] = 1
            cat_unc[c] = float(unc[i])
        else:
            n0 = cat_count[c]
            cat_mean[c] = unit((cat_mean[c] * n0 + h[i]) / float(n0 + 1))
            cat_count[c] += 1
            cat_unc[c] = (cat_unc[c] * n0 + float(unc[i])) / float(n0 + 1)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return np.zeros(8, np.float32), np.float32(-1.0), X, y
    clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced",
                             random_state=2717)
    clf.fit(X, y)
    return clf.coef_[0].astype(np.float32), np.float32(clf.intercept_[0]), X, y


def atomic_save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--novel-weight", type=float, default=5.0)
    ap.add_argument("--known-weight", type=float, default=5.0,
                    help="positive weight for the binary KNOWN evidence head")
    ap.add_argument("--legacy-three-way", action="store_true",
                    help="historical ablation only; not the Phase 9A head")
    ap.add_argument("--exclude-fp-negatives", action="store_true",
                    help="maturity ablation: omit false-birth negatives")
    ap.add_argument("--visible-only", action="store_true",
                    help="debug ablation; default uses all frozen known anchors")
    ap.add_argument("--dual-known-output", action="store_true",
                    help="use visible classes for knownness but all anchors for output IDs")
    ap.add_argument("--out", default="outputs/iclr27_phase9a/training/lifecycle")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    ep = load_episode("train_hard")
    split = json.loads((ROOT / "outputs/iclr27_phase7c/assets/class_split_hard.json").read_text())
    visible = [int(c) for c in split["train_visible"]]
    h_raw = load_foundation(len(ep["gt_role"]))
    h, age, unc, con = causal_tracks(ep, h_raw)
    visible_protos = known_prototypes(ep, h_raw, visible)
    if args.visible_only:
        protos, proto_ids = visible_protos, visible
        decision_protos, decision_ids = protos, proto_ids
    elif args.dual_known_output:
        protos, proto_ids = foundation_known_prototypes()
        decision_protos, decision_ids = visible_protos, visible
    else:
        protos, proto_ids = foundation_known_prototypes()
        decision_protos, decision_ids = protos, proto_ids
    X = action_matrix(ep, h, age, unc, con, decision_protos)
    y = np.where(ep["row_split"] == 0, 0,
                 np.where(ep["row_split"] == 1, 1, 2)).astype(np.int64)
    if args.legacy_three_way:
        action_w, action_b = fit_action_legacy(X, y, args.novel_weight)
        action_mode = "legacy_three_way_ablation"
    else:
        action_w, action_b = fit_action_binary(X, y == 0, args.known_weight)
        action_mode = "binary_knownness"
    maturity_w, maturity_b, Xm, ym = fit_maturity(
        ep, h, age, unc, con, decision_protos, visible,
        include_fp_negatives=not args.exclude_fp_negatives)
    reuse_w, reuse_b, Xr, yr = fit_reuse(ep, h, age, unc, decision_protos)
    atomic_save_npz(out / "heads.npz", action_w=action_w, action_b=action_b,
                    maturity_w=maturity_w, maturity_b=maturity_b,
                    reuse_w=reuse_w, reuse_b=reuse_b)
    atomic_save_npz(out / "known_prototypes.npz", prototypes=protos,
                    known_ids=np.asarray(proto_ids, dtype=np.int64))
    atomic_save_npz(out / "decision_prototypes.npz", prototypes=decision_protos,
                    known_ids=np.asarray(decision_ids, dtype=np.int64))
    meta = {
        "protocol": "Phase9A_legal_hard_train",
        "foundation": "outputs/iclr27_phase7c/assets/h_all.npz",
        "n_rows": int(len(ep["gt_role"])),
        "n_visible_known": len(visible),
        "n_known_prototypes": len(proto_ids),
        "n_decision_prototypes": len(decision_ids),
        "dual_known_output": bool(args.dual_known_output),
        "action_class_counts": {str(i): int(np.sum(y == i)) for i in range(3)},
        "maturity_examples": int(len(ym)),
        "maturity_positive_rate": float(np.mean(ym)) if len(ym) else 0.0,
        "reuse_examples": int(len(yr)),
        "reuse_positive_rate": float(np.mean(yr)) if len(yr) else 0.0,
        "novel_weight": float(args.novel_weight),
        "known_weight": float(args.known_weight),
        "action_mode": action_mode,
        "include_fp_negatives": bool(not args.exclude_fp_negatives),
    }
    tmp = out / "metadata.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, out / "metadata.json")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
