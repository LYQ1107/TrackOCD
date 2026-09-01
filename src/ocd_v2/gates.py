"""Known/novel routing gates (CLIP-only, DINO-only, dual-space) with proxy
calibration on train-known classes."""
from __future__ import annotations

import itertools
import math

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from src.ocd_v2.common import build_prototypes, proxy_split


def gate_scores(x, protos):
    best, best_id = -1.0, None
    for cid, p in protos.items():
        s = float(np.dot(x, p))
        if s > best:
            best, best_id = s, cid
    return best, best_id


class KnownGate:
    def __init__(self, protos, w=1.0, thr=0.5):
        self.protos = protos
        self.w = w
        self.thr = thr
        self.name = "gate"

    def score(self, x_clip, x_dino):
        s_clip, _ = gate_scores(x_clip, self.protos)
        return s_clip

    def decide(self, x_clip, x_dino, meta=None):
        s, cid = gate_scores(x_clip, self.protos)
        return s >= self.thr, cid, s


class DinoGate(KnownGate):
    def __init__(self, protos, thr=0.5):
        super().__init__(protos, w=1.0, thr=thr)
        self.name = "dino_gate"

    def score(self, x_clip, x_dino):
        s, _ = gate_scores(x_dino, self.protos)
        return s

    def decide(self, x_clip, x_dino, meta=None):
        s, cid = gate_scores(x_dino, self.protos)
        return s >= self.thr, cid, s


class DualGate:
    def __init__(self, clip_protos, dino_protos, w=0.5, thr=0.55):
        self.clip_protos = clip_protos
        self.dino_protos = dino_protos
        self.w = w
        self.thr = thr
        self.name = "dual_gate"

    def score(self, x_clip, x_dino):
        s_clip, cid_clip = gate_scores(x_clip, self.clip_protos)
        s_dino, cid_dino = gate_scores(x_dino, self.dino_protos)
        return self.w * s_clip + (1 - self.w) * s_dino, (s_clip, s_dino, cid_clip, cid_dino)

    def decide(self, x_clip, x_dino, meta=None):
        s_clip, cid_clip = gate_scores(x_clip, self.clip_protos)
        s_dino, cid_dino = gate_scores(x_dino, self.dino_protos)
        s = self.w * s_clip + (1 - self.w) * s_dino
        cid = cid_clip if self.w * s_clip >= (1 - self.w) * s_dino else cid_dino
        return s >= self.thr, cid, s


def top2_sims(x, protos):
    vals = np.asarray([float(np.dot(x, p)) for p in protos.values()])
    if len(vals) == 0:
        return -1.0, -1.0
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    order = np.argsort(vals)[::-1]
    return float(vals[order[0]]), float(vals[order[1]])


def gate_features(x_clip, x_dino, clip_protos, dino_protos, meta=None):
    d_best, d_sec = top2_sims(x_dino, dino_protos)
    c_best, c_sec = top2_sims(x_clip, clip_protos)
    n_frames = meta.get("num_frames", 1) if meta else 1
    mean_area = meta.get("mean_area", 0.0) if meta else 0.0
    return np.asarray(
        [
            d_best,
            d_sec,
            d_best - d_sec,
            c_best,
            c_sec,
            c_best - c_sec,
            math.log(n_frames + 1.0),
            math.log(mean_area + 1.0),
        ],
        dtype=np.float64,
    )


class GateLR:
    """G3 dual-space logistic-regression gate.
    Features: DINO/CLIP top-1/top-2 similarity, margins, track length, area."""

    name = "dual_lr_gate"

    def __init__(self, clip_protos, dino_protos, coef, intercept, thr):
        self.clip_protos = clip_protos
        self.dino_protos = dino_protos
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.thr = float(thr)

    def score(self, x_clip, x_dino, meta=None):
        f = gate_features(x_clip, x_dino, self.clip_protos, self.dino_protos, meta)
        z = float(self.coef @ f + self.intercept)
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def decide(self, x_clip, x_dino, meta=None):
        p = self.score(x_clip, x_dino, meta)
        _, cid = gate_scores(x_dino, self.dino_protos)
        return p >= self.thr, cid, p


class KnownMatcher:
    """Known-class matcher: CLIP+DINO weighted cosine nearest prototype.
    Separate from the known/novel router; w chosen on train-known labels."""

    def __init__(self, clip_protos, dino_protos, w=0.5):
        self.clip_protos = clip_protos
        self.dino_protos = dino_protos
        self.w = w

    def classify(self, x_clip, x_dino):
        best_cid, best_s = None, -1.0
        for cid in self.dino_protos:
            s = self.w * float(np.dot(x_clip, self.clip_protos[cid])) + (
                1.0 - self.w
            ) * float(np.dot(x_dino, self.dino_protos[cid]))
            if s > best_s:
                best_s, best_cid = s, cid
        return best_cid, best_s


def gate_metrics(scores, labels):
    labels = np.asarray(labels, dtype=int)
    preds = (np.asarray(scores) >= 0.0).astype(int)
    return preds


def calibrate_gate(clip_feats, dino_feats, labels, gate_type="dual", meta=None):
    """Calibrate on proxy-known (known) vs proxy-novel (novel) train classes.
    Returns (best_params, best_metrics, proxy_known, proxy_novel)."""
    proxy_known, proxy_novel = proxy_split(labels, seed=1027)
    ids = sorted(s for s in labels if s in clip_feats and s in dino_feats)
    y = np.array([1 if labels[s] in proxy_known else 0 for s in ids])
    known_ids = [s for s in ids if labels[s] in proxy_known]
    novel_ids = [s for s in ids if labels[s] in proxy_novel]
    clip_protos = build_prototypes(clip_feats, labels, proxy_known)
    dino_protos = build_prototypes(dino_feats, labels, proxy_known)

    def evaluate(gate):
        scores = []
        preds = []
        for s in ids:
            xc = clip_feats[s]
            xd = dino_feats[s]
            if gate_type == "dual":
                sc, _ = gate.score(xc, xd)
            else:
                sc = gate.score(xc, xd)
            scores.append(sc)
            is_k, _, _ = gate.decide(xc, xd)
            preds.append(1 if is_k else 0)
        scores = np.asarray(scores)
        preds = np.asarray(preds)
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        tn = ((preds == 0) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        known_rec = tp / (tp + fn) if tp + fn else 0.0
        known_prec = tp / (tp + fp) if tp + fp else 0.0
        novel_rec = tn / (tn + fp) if tn + fp else 0.0
        novel_prec = tn / (tn + fn) if tn + fn else 0.0
        auroc = roc_auc_score(y, scores) if len(set(y)) > 1 else 0.0
        return {
            "known_recall": known_rec,
            "known_precision": known_prec,
            "novel_recall": novel_rec,
            "novel_precision": novel_prec,
            "auroc": auroc,
            "false_known_rate": fp / (fp + tn) if fp + tn else 0.0,
            "false_novel_rate": fn / (fn + tp) if fn + tp else 0.0,
            "balanced": 0.5 * (known_rec + novel_rec),
        }

    best = None
    best_params = None
    if gate_type == "clip":
        for thr in np.arange(0.30, 0.86, 0.025):
            gate = KnownGate(clip_protos, thr=float(thr))
            m = evaluate(gate)
            if best is None or m["balanced"] > best["balanced"]:
                best, best_params = m, {"thr": float(thr)}
    elif gate_type == "dino":
        for thr in np.arange(0.30, 0.86, 0.025):
            gate = DinoGate(dino_protos, thr=float(thr))
            m = evaluate(gate)
            if best is None or m["balanced"] > best["balanced"]:
                best, best_params = m, {"thr": float(thr)}
    elif gate_type == "dual":
        for w in np.arange(0.0, 1.01, 0.1):
            for thr in np.arange(0.35, 0.91, 0.025):
                gate = DualGate(clip_protos, dino_protos, w=float(w), thr=float(thr))
                m = evaluate(gate)
                if best is None or m["balanced"] > best["balanced"]:
                    best, best_params = m, {"w": float(w), "thr": float(thr)}
    elif gate_type == "dual_lr":
        X = np.stack(
            [
                gate_features(clip_feats[s], dino_feats[s], clip_protos, dino_protos, meta.get(s) if meta else None)
                for s in ids
            ]
        )
        lr = LogisticRegression(max_iter=3000).fit(X, y)
        p = lr.predict_proba(X)[:, 1]
        best = None
        best_params = None
        for thr in np.arange(0.05, 0.96, 0.025):
            preds = (p >= thr).astype(int)
            tp = ((preds == 1) & (y == 1)).sum()
            fp = ((preds == 1) & (y == 0)).sum()
            tn = ((preds == 0) & (y == 0)).sum()
            fn = ((preds == 0) & (y == 1)).sum()
            kr = tp / (tp + fn) if tp + fn else 0.0
            nr = tn / (tn + fp) if tn + fp else 0.0
            m = {
                "known_recall": kr,
                "known_precision": tp / (tp + fp) if tp + fp else 0.0,
                "novel_recall": nr,
                "novel_precision": tn / (tn + fn) if tn + fn else 0.0,
                "auroc": roc_auc_score(y, p) if len(set(y)) > 1 else 0.0,
                "false_known_rate": fp / (fp + tn) if fp + tn else 0.0,
                "false_novel_rate": fn / (fn + tp) if fn + tp else 0.0,
                "balanced": 0.5 * (kr + nr),
            }
            if best is None or m["balanced"] > best["balanced"]:
                best = m
                best_params = {
                    "coef": [float(v) for v in lr.coef_[0]],
                    "intercept": float(lr.intercept_[0]),
                    "thr": float(thr),
                }
    else:
        raise ValueError(gate_type)
    return best_params, best, proxy_known, proxy_novel


def build_gate(gate_type, clip_feats, labels, dino_feats=None, params=None):
    if gate_type == "clip":
        protos = build_prototypes(clip_feats, labels, set(labels.values()))
        return KnownGate(protos, thr=params["thr"])
    if gate_type == "dino":
        protos = build_prototypes(dino_feats, labels, set(labels.values()))
        return DinoGate(protos, thr=params["thr"])
    if gate_type == "dual_lr":
        clip_protos = build_prototypes(clip_feats, labels, set(labels.values()))
        dino_protos = build_prototypes(dino_feats, labels, set(labels.values()))
        return GateLR(
            clip_protos,
            dino_protos,
            coef=params["coef"],
            intercept=params["intercept"],
            thr=params["thr"],
        )
    clip_protos = build_prototypes(clip_feats, labels, set(labels.values()))
    dino_protos = build_prototypes(dino_feats, labels, set(labels.values()))
    return DualGate(clip_protos, dino_protos, w=params["w"], thr=params["thr"])
