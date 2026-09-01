"""Legal memory-state features for ORBIT-MSRouting (Phase 4G).

Every feature is computable at decision time from the causal memory state
produced by the model's own previous decisions:

  log_mem                  log1p(memory size)
  mean_support             log1p(mean prototype support)
  low_support_ratio        share of prototypes with support <= 2
  mean_dispersion          mean running dispersion (clipped)
  high_disp_ratio          share of prototypes with dispersion > 0.3
  recent_birth_rate        NEW_NOVEL share in the last `window` decisions
  recent_reuse_rate        EXISTING_NOVEL share in the last `window`
  low_support_birth_proxy  share of recent births whose prototype still has
                           support <= 2 (legal proxy for provisional /
                           known-origin contamination; GT is never used)

No GT purity, no true novel count, no final memory size, no future support.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

STATE_FEAT_ORDER = [
    "log_mem",
    "mean_support",
    "low_support_ratio",
    "mean_dispersion",
    "high_disp_ratio",
    "recent_birth_rate",
    "recent_reuse_rate",
    "low_support_birth_proxy",
]


def log1p_norm(x, cap=300.0):
    return math.log1p(max(float(x), 0.0)) / math.log1p(cap)


class MemoryStateTracker:
    """Tracks the legal memory-state statistics and a recent-action window.

    The window is a deque of (action, vid) records, where action is one of
    "KNOWN", "EXISTING_NOVEL", "NEW_NOVEL".  All statistics are computed
    before the current decision is applied.
    """

    def __init__(self, window=32):
        self.window = window
        self.recent = deque(maxlen=window)
        self.birth_vids = {}

    def note_action(self, action, vid=None):
        self.recent.append((action, vid))
        if action == "NEW_NOVEL" and vid is not None:
            self.birth_vids[vid] = len(self.recent) - 1

    def compute(self, mem, feat_names=None):
        names = feat_names or STATE_FEAT_ORDER
        n = len(mem.novel)
        if n == 0:
            supports = []
            dispersions = []
        else:
            supports = [mem.support(v) for v in mem.novel]
            dispersions = [float(mem.meta[v]["dispersion"])
                           for v in mem.novel]
        mean_s = float(sum(supports)) / max(n, 1)
        low_support_ratio = (sum(1 for s in supports if s <= 2)
                             / max(n, 1))
        mean_disp = (float(sum(dispersions)) / max(n, 1)
                     if dispersions else 0.0)
        high_disp_ratio = (sum(1 for d in dispersions if d > 0.3)
                           / max(n, 1))
        wlen = len(self.recent)
        births = [a for a, _ in self.recent if a == "NEW_NOVEL"]
        reuses = [a for a, _ in self.recent if a == "EXISTING_NOVEL"]
        recent_birth_rate = len(births) / max(wlen, 1)
        recent_reuse_rate = len(reuses) / max(wlen, 1)
        low_support_births = 0
        for a, vid in self.recent:
            if a == "NEW_NOVEL" and vid is not None:
                if vid in mem.novel and mem.support(vid) <= 2:
                    low_support_births += 1
        low_support_birth_proxy = (
            low_support_births / max(len(births), 1))
        vals = {
            "log_mem": log1p_norm(n),
            "mean_support": log1p_norm(mean_s),
            "low_support_ratio": min(max(low_support_ratio, 0.0), 1.0),
            "mean_dispersion": min(max(mean_disp, 0.0), 1.0),
            "high_disp_ratio": min(max(high_disp_ratio, 0.0), 1.0),
            "recent_birth_rate": min(max(recent_birth_rate, 0.0), 1.0),
            "recent_reuse_rate": min(max(recent_reuse_rate, 0.0), 1.0),
            "low_support_birth_proxy": min(
                max(low_support_birth_proxy, 0.0), 1.0),
        }
        return [vals[nm] for nm in names]

    def summary(self, mem):
        """Small dict used for logging (all legal)."""
        names = STATE_FEAT_ORDER
        vals = self.compute(mem, names)
        return dict(zip(names, vals))
