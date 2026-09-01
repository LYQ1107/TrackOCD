"""Causal known-novel memory for ORBIT-FC.

Extends the ORBIT BiMemory with prototype creation timestamps and optional
support-aware radius updates.  The online causal contract is unchanged:
tracks are processed one at a time, history is never rewritten, and the
number of novel classes is never supplied.
"""
from __future__ import annotations

import numpy as np

from src.orbit.bi_memory import BiMemory


class CausalNovelMemory(BiMemory):
    def __init__(self, known_protos, known_radii=None, novel_update_rate=0.2,
                 radius_percentile=50.0, radius_update_rate=0.2):
        super().__init__(known_protos, known_radii, novel_update_rate)
        self.radius_percentile = radius_percentile
        self.radius_update_rate = radius_update_rate

    def create_novel(self, z: np.ndarray, created_at: int = 0) -> int:
        vid = super().create_novel(z)
        self.novel[vid]["created_at"] = created_at
        return vid

    def update_novel(self, vid: int, z: np.ndarray, cos_to_center: float = None,
                     update_radius: bool = False):
        c = self.novel[vid]
        n = self.novel_counts.get(vid, 1)
        w = self.novel_update_rate
        c["proto"] = self._norm((1 - w) * c["proto"] + w * z)
        self.novel_counts[vid] = n + 1
        if update_radius and cos_to_center is not None:
            self.novel_radii[vid] = float(
                (1 - self.radius_update_rate) * self.novel_radii.get(vid, 0.3)
                + self.radius_update_rate * max(1.0 - cos_to_center, 1e-3)
            )

    def support(self, vid: int) -> int:
        return int(self.novel_counts.get(vid, 0))

    def age(self, vid: int, current_index: int) -> int:
        created = int(self.novel.get(vid, {}).get("created_at", 0))
        return max(current_index - created, 0)
