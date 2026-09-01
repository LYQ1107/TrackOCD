"""ORBIT-IAM model: ORBIT-MSR + small pairwise identity compatibility head."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.orbit_fc.model import ORBITFCModel


class IdentityCompatibility(nn.Module):
    """Small interpretable MLP mapping pair statistics to compatibility.

    Inputs (ordered, subset configurable):
      sim, margin, radius, support_norm, conf, mem_scale, reliability
    """

    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ORBITIAMModel(ORBITFCModel):
    def __init__(self, dim=768, bottleneck=128, gate_dim=11, reuse_dim=11,
                 hidden=64, use_adapter=True, compat_dim=6, state_dim=0):
        super().__init__(dim, bottleneck, gate_dim, reuse_dim, hidden,
                         use_adapter)
        self.compat = IdentityCompatibility(compat_dim)
        self.state_bias = None
        if state_dim > 0:
            self.state_bias = nn.Sequential(
                nn.Linear(state_dim, 8), nn.ReLU(), nn.Linear(8, 1),
            )

    def compat_forward(self, x):
        return self.compat(x)

    def gate_logit_with_bias(self, logit, state_feats):
        if self.state_bias is None:
            return logit
        b = self.state_bias(state_feats)
        return logit - b
