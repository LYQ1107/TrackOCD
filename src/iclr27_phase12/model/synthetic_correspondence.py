"""Small causal trajectory encoder for legal synthetic OOD episodes."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SyntheticTrajectoryEncoder(nn.Module):
    """Causal GRU metric encoder; physical IDs never enter the network."""

    def __init__(self, in_dim: int = 128, hidden: int = 128, out_dim: int = 128):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)
        self.gru = nn.GRU(self.in_dim, self.hidden, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden + self.in_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.out_dim),
        )

    def new_state(self, batch: int, device):
        return torch.zeros(1, batch, self.hidden, device=device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        b, t, _ = x.shape
        if mask is None:
            mask = torch.ones(b, t, dtype=torch.bool, device=x.device)
        else:
            mask = mask.bool()
        out, _ = self.gru(x)
        last = torch.clamp(mask.long().sum(1) - 1, min=0)
        cur = x[torch.arange(b, device=x.device), last]
        dyn = out[torch.arange(b, device=x.device), last]
        z = F.normalize(self.proj(torch.cat([dyn, cur], dim=-1)), dim=-1)
        return z, out
