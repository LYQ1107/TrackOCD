"""Causal track-level semantic encoder for Phase 13."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackSemanticEncoder(nn.Module):
    """Appearance + motion GRU with no physical-ID or semantic memory input."""

    def __init__(self, appearance_dim: int = 768, motion_dim: int = 4,
                 hidden: int = 128, out_dim: int = 128):
        super().__init__()
        self.appearance_dim = int(appearance_dim)
        self.motion_dim = int(motion_dim)
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)
        self.appearance = nn.Sequential(
            nn.LayerNorm(self.appearance_dim),
            nn.Linear(self.appearance_dim, self.hidden),
            nn.GELU(),
        )
        self.motion = nn.Sequential(
            nn.LayerNorm(self.motion_dim),
            nn.Linear(self.motion_dim, 32),
            nn.GELU(),
        )
        self.gru = nn.GRU(self.hidden + 32, self.hidden, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden + self.hidden + 32, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.out_dim),
        )
        self.gate = nn.Linear(self.hidden + self.hidden + 32, self.out_dim)
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def new_state(self, batch: int = 1, device=None):
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(1, batch, self.hidden, device=device)
        prev_a = torch.zeros(batch, self.hidden, device=device)
        last = torch.zeros(batch, self.out_dim, device=device)
        return h, prev_a, last

    def step(self, appearance: torch.Tensor, motion: torch.Tensor, state):
        a = self.appearance(appearance)
        m = self.motion(motion)
        inp = torch.cat([a, m], dim=-1)
        h0, prev_a, _ = state
        seq, h1 = self.gru(inp.unsqueeze(1), h0)
        dyn = seq[:, 0]
        delta = a - prev_a
        f = torch.cat([dyn, a, m], dim=-1)
        y = F.normalize(a + torch.sigmoid(self.gate(f)) * self.fusion(f), dim=-1)
        return y, (h1, a, y)

    def forward(self, appearance: torch.Tensor, motion: torch.Tensor,
                mask: torch.Tensor | None = None):
        b, t, _ = appearance.shape
        if mask is None:
            mask = torch.ones(b, t, dtype=torch.bool, device=appearance.device)
        else:
            mask = mask.bool()
        h, prev_a, last = self.new_state(b, appearance.device)
        outs = []
        for i in range(t):
            y, (h1, pa1, _) = self.step(appearance[:, i], motion[:, i],
                                         (h, prev_a, last))
            valid = mask[:, i]
            h = torch.where(valid.view(1, b, 1), h1, h)
            prev_a = torch.where(valid.view(b, 1), pa1, prev_a)
            last = torch.where(valid.view(b, 1), y, last)
            outs.append(last)
        seq = torch.stack(outs, dim=1)
        lengths = torch.clamp(mask.long().sum(1) - 1, min=0)
        final = seq[torch.arange(b, device=appearance.device), lengths]
        return F.normalize(final, dim=-1), F.normalize(seq, dim=-1)
