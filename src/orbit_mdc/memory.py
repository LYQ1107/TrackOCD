"""Memory-Dynamics-Consistent memory for ORBIT-MDC.

Extends the Phase 4E IamMemory with the legal state bookkeeping needed for
on-policy rollout training:

* prototype identity: the pseudo-label of the track that *created* the
  prototype (set once at birth; never changed by later, possibly wrong,
  assignments).  This is a training-time notion only; it is never exposed to
  the decision policy and never used on the official stream.
* optional provisional-influence (quarantine) scaling, applied at decision
  time to the compatibility score.  Virtual IDs are assigned at birth and
  historical labels are never rewritten.

All statistics remain causal (updated only from accepted arrivals before the
current decision).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from src.orbit_iam.iam_memory import IamMemory


class MdcMemory(IamMemory):
    """IamMemory + identity map + optional provisional-influence factor."""

    def __init__(self, known_protos, known_radii=None, novel_update_rate=0.2,
                 radius_percentile=50.0, radius_update_rate=0.2,
                 quarantine_mode=0, quarantine_support_thr=3,
                 quarantine_dispersion_thr=0.3, quarantine_coef=1.0):
        super().__init__(known_protos, known_radii, novel_update_rate,
                         radius_percentile, radius_update_rate)
        self.quarantine_mode = quarantine_mode
        self.quarantine_support_thr = max(int(quarantine_support_thr), 2)
        self.quarantine_dispersion_thr = float(quarantine_dispersion_thr)
        self.quarantine_coef = float(quarantine_coef)
        self.identity = {}  # vid -> pseudo-label of the creating track
        self.vids_by_identity = defaultdict(list)

    def create_novel(self, z, created_at=0, identity=None):
        vid = super().create_novel(z, created_at)
        self.identity[vid] = identity
        if identity is not None:
            self.vids_by_identity[identity].append(vid)
        return vid

    def update_novel(self, vid, z, cos_to_center=None, update_radius=False,
                     margin=None):
        super().update_novel(vid, z, cos_to_center=cos_to_center,
                             update_radius=update_radius, margin=margin)

    def influence(self, vid):
        """Provisional-influence factor in [0, 1] at decision time.

        Q0: always 1.
        Q1: bounded until the prototype accumulates stable support.
        Q2: Q1 + dispersion penalty.

        Virtual IDs are never delayed; only the attraction strength is
        bounded.  Historical predictions are never rewritten.
        """
        if self.quarantine_mode == 0:
            return 1.0
        w = 1.0
        if self.quarantine_mode >= 1:
            s = self.support(vid)
            if s < self.quarantine_support_thr:
                w *= max((s - 1) / max(self.quarantine_support_thr - 1, 1),
                         0.0)
        if self.quarantine_mode >= 2:
            m = self.meta[vid]
            disp = float(m["dispersion"])
            w *= float(math.exp(-max(disp - self.quarantine_dispersion_thr,
                                     0.0) / 0.3))
        if self.quarantine_coef <= 0.0:
            return 1.0
        return 1.0 - self.quarantine_coef * (1.0 - w)

    def state(self, vid, arrival=0):
        st = super().state(vid)
        st["influence"] = self.influence(vid)
        return st
