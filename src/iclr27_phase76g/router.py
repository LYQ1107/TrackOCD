from __future__ import annotations

import torch
from torch import nn


class GroupRobustRelationRouter(nn.Module):
    """Small HELP/HARM/NEUTRAL router trained with rotating group holdouts."""

    input_dim = 14
    classes = ("HELP", "HARM", "NEUTRAL")

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 3)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"router features must be [N,{self.input_dim}], got {tuple(features.shape)}")
        return self.net(features)
