from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

ROUTER_NAMES = ["R0", "R1", "R2", "R3", "R4", "R5"]


FEATURE_SETS = {
    "R2": ["s1", "margin", "z_top1", "entropy"],
    "R3": ["s1", "margin", "k1", "k5", "k10", "nearest_dist"],
    "R4": ["s1", "s2", "margin", "z_top1", "entropy", "k1", "k5", "k10",
           "proto_consistency", "log_len", "frame_consistency", "log_area"],
}


def make_router(name, protos, threshold=0.45, margin_thr=0.0, coef=None,
                intercept=None, feat_names=None):
    if name == "R0":
        return LegacyRouter(protos, threshold)
    if name == "R1":
        return MarginRouter(protos, threshold, margin_thr)
    return LogisticRouter(protos, FEATURE_SETS[name], threshold,
                          coef=coef, intercept=intercept, feat_names=feat_names)


class LegacyRouter:
    def __init__(self, protos, threshold):
        self.protos = protos
        self.threshold = threshold

    def score(self, feats):
        return feats["s1"]

    def decide(self, feats):
        return self.score(feats) >= self.threshold


class MarginRouter:
    def __init__(self, protos, threshold, margin_thr):
        self.protos = protos
        self.threshold = threshold
        self.margin_thr = margin_thr

    def decide(self, feats):
        return feats["s1"] >= self.threshold and feats["margin"] >= self.margin_thr


class LogisticRouter:
    def __init__(self, protos, feature_names, threshold, coef=None,
                 intercept=None, feat_names=None):
        self.protos = protos
        self.feature_names = feature_names
        self.threshold = threshold
        self.coef = np.asarray(coef, dtype=np.float64) if coef is not None else None
        self.intercept = float(intercept) if intercept is not None else None

    def score(self, feats):
        v = np.array([feats[f] for f in self.feature_names])
        z = float(self.coef @ v + self.intercept)
        return 1.0 / (1.0 + np.exp(-z))

    def decide(self, feats):
        return self.score(feats) >= self.threshold
