"""Causal novel memory with legal per-prototype identity statistics.

Extends ``CausalNovelMemory`` with the statistics used by the IAM
pairwise identity-compatibility head and by prototype confidence:

* support count,
* within-cluster cosine dispersion (running mean of 1 - cos to center),
* accepted-assignment margin history (mean/min/low-margin count/recent std),
* prototype age.

All statistics are updated strictly causally from the accepted stream and
never use GT labels.  The confidence value is an online-computable scalar:

    c = log1p(support)/log1p(20) * exp(-dispersion/0.3)
        * (1 - low_margin_rate) * margin_stability

This mirrors the Phase 4E prototype-confidence audit, with the GT-derived
assignment-consistency component replaced by margin stability (legal).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from src.orbit_fc.causal_memory import CausalNovelMemory


class IamMemory(CausalNovelMemory):
    def __init__(self, known_protos, known_radii=None, novel_update_rate=0.2,
                 radius_percentile=50.0, radius_update_rate=0.2,
                 recent_window=12, low_margin_thr=0.02):
        super().__init__(known_protos, known_radii, novel_update_rate,
                         radius_percentile, radius_update_rate)
        self.recent_window = recent_window
        self.low_margin_thr = low_margin_thr
        self.meta = defaultdict(lambda: {
            "created_at": None,
            "dispersion": 0.0,
            "disp_n": 0,
            "mean_margin": 0.0,
            "margin_n": 0,
            "min_margin": 1.0,
            "low_margin_count": 0,
            "recent_margins": [],
        })

    def create_novel(self, z: np.ndarray, created_at: int = 0) -> int:
        vid = super().create_novel(z, created_at)
        self.meta[vid]["created_at"] = created_at
        return vid

    def update_novel(self, vid: int, z: np.ndarray, cos_to_center: float = None,
                     update_radius: bool = False, margin: float = None):
        m = self.meta[vid]
        if cos_to_center is None:
            cos_to_center = float(np.dot(self.novel[vid]["proto"], z))
        d = max(1.0 - cos_to_center, 0.0)
        m["disp_n"] += 1
        m["dispersion"] = ((m["disp_n"] - 1) * m["dispersion"] + d) / m["disp_n"]
        if margin is not None:
            n = m["margin_n"] + 1
            m["mean_margin"] = (m["mean_margin"] * m["margin_n"] + margin) / n
            m["margin_n"] = n
            m["min_margin"] = min(m["min_margin"], margin)
            if margin < self.low_margin_thr:
                m["low_margin_count"] += 1
            m["recent_margins"].append(margin)
            if len(m["recent_margins"]) > self.recent_window:
                m["recent_margins"] = m["recent_margins"][-self.recent_window:]
        super().update_novel(vid, z, cos_to_center=cos_to_center,
                             update_radius=update_radius)

    def margin_stability(self, vid: int) -> float:
        m = self.meta[vid]
        recent = m["recent_margins"]
        if len(recent) < 2:
            return 1.0
        std = float(np.std(recent))
        return float(math.exp(-min(std, 0.2) / 0.1))

    def confidence(self, vid: int) -> float:
        m = self.meta[vid]
        low_rate = m["low_margin_count"] / max(m["margin_n"], 1)
        return (math.log1p(self.support(vid)) / math.log1p(20.0)
                * math.exp(-m["dispersion"] / 0.3)
                * (1.0 - low_rate) * self.margin_stability(vid))

    def state(self, vid: int, arrival: int) -> dict:
        """Decision-time state of prototype vid (before current assignment)."""
        m = self.meta[vid]
        return {
            "radius": float(self.novel_radii.get(vid, 0.3)),
            "support": self.support(vid),
            "dispersion": float(m["dispersion"]),
            "age": max(arrival - (m["created_at"] or 0), 0),
            "mean_margin": float(m["mean_margin"]),
            "min_margin": float(m["min_margin"]),
            "low_margin_count": int(m["low_margin_count"]),
            "margin_stability": self.margin_stability(vid),
            "confidence": self.confidence(vid),
        }
