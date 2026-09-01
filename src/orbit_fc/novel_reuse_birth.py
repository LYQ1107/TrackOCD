"""Novel Reuse-Birth Decision (binary EXISTING_NOVEL vs NEW_NOVEL)."""
from __future__ import annotations

import torch
import torch.nn as nn


class NovelReuseBirth(nn.Module):
    """Binary head trained only on non-known tracks."""

    def __init__(self, stats_dim: int = 13, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stats_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.net(stats).squeeze(-1)
