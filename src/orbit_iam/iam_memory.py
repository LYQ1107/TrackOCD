"""Identity-Aware Matching memory: CausalNovelMemory + online statistics.

All statistics are computable online (support, radius, dispersion, mean /
minimum accepted margin, low-margin rate, age).  GT purity is never used
here; offline purity is attached only in audit scripts.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from src.orbit_fc.causal_memory import CausalNovelMemory


class IamMemory(CausalNovelMemory):
    def __init__(self, known_protos, known_radii=None, novel_update_rate=0.2,
                 radius_percentile=50.0, radius_update_rate=0.2):
        super().__init__(known_protos, known_radii, novel_update_rate,
                         radius_percentile, radius_update_rate)
        self.meta = defaultdict(lambda: {
            "created_at": None,
            "dispersion": 0.3,
            "disp_n": 0,
            "mean_margin": 0.0,
            "min_margin": 1.0,
            "low_margin_count": 0,
            "margin_n": 0,
        })

    def create_novel(self, z, created_at=0):
        vid = super().create_novel(z, created_at)
        self.meta[vid]["created_at"] = created_at
        return vid

    def update_novel(self, vid, z, cos_to_center=None, update_radius=False,
                     margin=None):
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
            if margin < 0.02:
                m["low_margin_count"] += 1
        super().update_novel(vid, z, cos_to_center=cos_to_center,
                             update_radius=update_radius)

    def conf(self, vid):
        """Legal online confidence in [0, 1]."""
        m = self.meta[vid]
        support = math.log1p(self.support(vid)) / math.log1p(20.0)
        disp = math.exp(-m["dispersion"] / 0.3)
        low = 1.0 - m["low_margin_count"] / max(m["margin_n"], 1)
        return float(min(max(support * disp * low, 0.0), 1.0))

    def state(self, vid):
        m = self.meta[vid]
        return {
            "radius": float(self.novel_radii.get(vid, 0.3)),
            "support": self.support(vid),
            "dispersion": float(m["dispersion"]),
            "mean_margin": float(m["mean_margin"]),
            "min_margin": float(m["min_margin"]),
            "low_margin_rate": m["low_margin_count"] / max(m["margin_n"], 1),
            "age": max(int(m["created_at"] or 0), 0),
            "conf": self.conf(vid),
        }

    def state_summary(self):
        """Legal current-state statistics for state-conditioned routing."""
        n = len(self.novel)
        if n == 0:
            return {
                "log_mem": 0.0, "mean_support": 0.0,
                "low_support_ratio": 0.0, "mean_dispersion": 0.0,
            }
        supports = [self.support(v) for v in self.novel]
        disps = [self.meta[v]["dispersion"] for v in self.novel]
        return {
            "log_mem": math.log1p(n),
            "mean_support": float(np.mean(supports)),
            "low_support_ratio": sum(1 for s in supports if s <= 2) / n,
            "mean_dispersion": float(np.mean(disps)),
        }
