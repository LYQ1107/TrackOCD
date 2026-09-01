"""ORBIT-MDC model: ORBIT-IAM + birth/reuse decision head.

The birth head consumes relative evidence (best vs second-best compat
probability, margin, best prototype state, track reliability, memory scale)
and outputs a reuse logit.  It exists to replace the knife-edge absolute
compatibility threshold with a memory-state-conditioned reuse decision.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.orbit_iam.model import ORBITIAMModel


class BirthHead(nn.Module):
    """Small MLP: [q_best, q_second, q_margin, support, dispersion, rel, mem]
    -> reuse logit."""

    def __init__(self, in_dim=7, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


BIRTH_FEAT_ORDER = ["q_best", "q_second", "q_margin", "support", "dispersion",
                    "rel", "mem"]


def build_birth_features(q_best, q_second, best_support, best_dispersion,
                         rel, mem_size, feat_names=None):
    names = feat_names or BIRTH_FEAT_ORDER
    q_margin = max(q_best - q_second, -1.0)
    vals = {
        "q_best": float(q_best),
        "q_second": float(q_second),
        "q_margin": float(q_margin),
        "support": float(min(max(best_support, 0.0), 1.0)),
        "dispersion": float(min(max(best_dispersion, 0.0), 1.0)),
        "rel": float(rel),
        "mem": float(min(max(mem_size, 0.0), 1.0)),
    }
    return [vals[n] for n in names]


class ORBITMDCModel(ORBITIAMModel):
    def __init__(self, dim=768, bottleneck=128, gate_dim=11, reuse_dim=11,
                 hidden=64, use_adapter=True, compat_dim=6, birth_dim=0):
        super().__init__(dim, bottleneck, gate_dim, reuse_dim, hidden,
                         use_adapter, compat_dim)
        self.birth_dim = birth_dim
        self.birth = BirthHead(birth_dim) if birth_dim > 0 else None

    def birth_forward(self, x):
        if self.birth is None:
            raise RuntimeError("birth head not configured")
        return self.birth(x)
