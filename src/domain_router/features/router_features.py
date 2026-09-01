from __future__ import annotations

import math

import numpy as np
from scipy.special import softmax


def prototype_sims(x, protos):
    return np.array([float(np.dot(x, p)) for p in protos.values()])


def top2(sims):
    order = np.argsort(sims)[::-1]
    return float(sims[order[0]]), float(sims[order[1]] if len(order) > 1 else sims[order[0]])


def knn_features(x, index):
    if index is None or len(index) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    sims = index @ x
    sims = np.sort(sims)[::-1]
    k1 = float(sims[0])
    k5 = float(sims[:5].mean())
    k10 = float(sims[:10].mean())
    nearest_dist = float(1.0 - sims[0])
    return [k1, k5, k10, nearest_dist]


def compute_router_features(x, protos, knn_index=None, frame_feats=None, meta=None):
    """x: L2 track-mean; protos: known prototypes; returns feature dict."""
    sims = prototype_sims(x, protos)
    s1, s2 = top2(sims)
    margin = s1 - s2
    mean_s, std_s = float(sims.mean()), float(sims.std()) + 1e-9
    z_top1 = (s1 - mean_s) / std_s
    p = softmax(sims)
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    k1, k5, k10, ndist = knn_features(x, knn_index)
    proto_consistency = s1
    n_frames = meta.get("num_frames", 1) if meta else 1
    mean_area = meta.get("mean_area", 0.0) if meta else 0.0
    if frame_feats is not None and len(frame_feats) > 1:
        F = np.asarray(frame_feats, dtype=np.float32)
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        S = Fn @ Fn.T
        n = len(F)
        frame_consistency = float((S.sum() - n) / max(1, n * (n - 1)))
    else:
        frame_consistency = 1.0
    return {
        "s1": s1, "s2": s2, "margin": margin, "z_top1": z_top1,
        "entropy": entropy, "k1": k1, "k5": k5, "k10": k10,
        "nearest_dist": ndist, "proto_consistency": proto_consistency,
        "log_len": math.log(n_frames + 1.0), "frame_consistency": frame_consistency,
        "log_area": math.log(mean_area + 1.0),
    }
