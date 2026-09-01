"""ORBIT model: geometry-preserving aggregation + action network."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.orbit.action_router import ActionNetwork
from src.orbit.track_aggregator import ReliabilityAggregator


class ORBITModel(nn.Module):
    def __init__(self, dim: int = 768, bottleneck: int = 128,
                 use_adapter: bool = True, use_reliability: bool = True,
                 stats_dim: int = 18):
        super().__init__()
        self.use_adapter = use_adapter
        self.aggregator = ReliabilityAggregator(dim, bottleneck, use_reliability)
        self.action_net = ActionNetwork(stats_dim)

    def aggregate(self, x: torch.Tensor, mask: torch.Tensor):
        if self.use_adapter:
            return self.aggregator.aggregate(x, mask)
        # D0: frozen DINO track mean
        z = torch.nn.functional.normalize(x.mean(dim=1), dim=-1)
        z0 = z.detach()
        return {"z": z, "z0": z0, "weights": mask.float() / mask.sum(dim=1, keepdim=True).clamp(min=1),
                "y": x, "cos": torch.ones(x.shape[:2], device=x.device),
                "length": mask.sum(dim=1)}
