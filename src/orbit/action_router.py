"""Action network for the causal known-novel bi-memory competition."""
from __future__ import annotations

import torch
import torch.nn as nn

KNOWN = 0
EXISTING_NOVEL = 1
NEW_NOVEL = 2
ACTIONS = ("KNOWN", "EXISTING_NOVEL", "NEW_NOVEL")


class ActionNetwork(nn.Module):
    def __init__(self, stats_dim: int = 18, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stats_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.net(stats)
