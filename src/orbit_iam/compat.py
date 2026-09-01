"""Shared compatibility feature construction for ORBIT-IAM."""
from __future__ import annotations

import math

import numpy as np

FEAT_ORDER = ["sim", "margin", "radius", "support", "conf", "mem", "rel"]


def compat_feature_spec(feat_str):
    names = [f.strip() for f in feat_str.split(",") if f.strip()]
    for n in names:
        assert n in FEAT_ORDER, f"unknown compat feature {n}"
    return names


def build_compat_features(z, proto, radius, support, conf, mem_size, rel,
                          margin, feat_names):
    """Return a list of features in FEAT_ORDER, only enabled ones."""
    sim = float(np.dot(z, proto))
    vals = {
        "sim": sim,
        "margin": float(margin),
        "radius": float(min(radius, 1.0)),
        "support": math.log1p(max(support, 0)) / math.log1p(300.0),
        "conf": float(conf),
        "mem": math.log1p(max(mem_size, 0)) / math.log1p(300.0),
        "rel": float(rel),
    }
    return [vals[n] for n in feat_names]


def compat_matrix_for_track(z, protos, states, mem_size, rel, margin,
                            feat_names):
    rows = []
    for vid, p in protos.items():
        st = states[vid]
        rows.append(build_compat_features(
            z, p, st["radius"], st["support"], st["conf"], mem_size, rel,
            margin, feat_names))
    return np.asarray(rows, dtype=np.float32) if rows else np.empty((0, len(feat_names)), dtype=np.float32)
