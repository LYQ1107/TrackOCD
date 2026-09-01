"""ORBIT-FC model: shared adapter + factorized known gate and reuse head."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.orbit.track_aggregator import ReliabilityAggregator
from src.orbit_fc.known_gate import KnownGate
from src.orbit_fc.novel_reuse_birth import NovelReuseBirth


class ORBITFCModel(nn.Module):
    def __init__(self, dim: int = 768, bottleneck: int = 128,
                 gate_dim: int = 13, reuse_dim: int = 13, hidden: int = 64,
                 use_adapter: bool = True, compat_dim: int = 0):
        super().__init__()
        self.use_adapter = use_adapter
        self.aggregator = ReliabilityAggregator(dim, bottleneck, use_reliability=False)
        self.gate = KnownGate(gate_dim, hidden)
        self.reuse = NovelReuseBirth(reuse_dim, hidden)
        self.compat = (PairwiseIdentityCompatibility(compat_dim, hidden)
                       if compat_dim > 0 else None)

    def aggregate(self, x: torch.Tensor, mask: torch.Tensor):
        if not self.use_adapter:
            z = torch.nn.functional.normalize(x.mean(dim=1), dim=-1)
            z0 = z.detach()
            return {"z": z, "z0": z0, "weights": mask.float() / mask.sum(dim=1, keepdim=True).clamp(min=1),
                    "y": x, "cos": torch.ones(x.shape[:2], device=x.device),
                    "length": mask.sum(dim=1)}
        return self.aggregator.aggregate(x, mask)

    def gate_forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.gate(stats)

    def reuse_forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.reuse(stats)

    def compat_forward(self, stats: torch.Tensor) -> torch.Tensor:
        if self.compat is None:
            raise RuntimeError("compat head not present in this checkpoint")
        return self.compat(stats)


class PairwiseIdentityCompatibility(nn.Module):
    """Small calibration head for pairwise identity compatibility.

    Inputs are interpretable per-pair statistics (similarity, relative
    margin, prototype radius/support/confidence, memory scale, track
    reliability).  Design follows the Phase 4E protocol: a small MLP /
    logistic-style head, no large networks.
    """

    def __init__(self, stats_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stats_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.net(stats).squeeze(-1)
