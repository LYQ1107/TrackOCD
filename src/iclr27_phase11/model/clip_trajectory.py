"""Causal trajectory adapter over a frozen CLIP visual-language space."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipTrajectoryEncoder(nn.Module):
    """Map CLIP frame vectors to a causal 128-D semantic trajectory vector.

    The CLIP visual encoder is frozen.  This module sees only the current
    frame vector and prefix state; a physical ID is used outside the module
    only to select the private recurrent state.
    """
    def __init__(self, in_dim: int = 512, dim: int = 128,
                 hidden: int = 128, out_dim: int = 128):
        super().__init__()
        self.in_dim = int(in_dim)
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(self.in_dim, self.dim), nn.LayerNorm(self.dim), nn.GELU())
        self.gru = nn.GRU(self.dim, self.hidden, batch_first=True)
        fdim = self.hidden + self.dim + self.dim
        self.fusion = nn.Sequential(
            nn.Linear(fdim, self.hidden), nn.GELU(), nn.Linear(self.hidden, self.out_dim))
        self.gate = nn.Linear(fdim, self.out_dim)
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def new_state(self, batch: int = 1, device=None):
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(1, batch, self.hidden, device=device)
        prev = torch.zeros(batch, self.dim, device=device)
        last = torch.zeros(batch, self.out_dim, device=device)
        return h, prev, last

    def step(self, x: torch.Tensor, state):
        x = self.input_proj(x)
        h0, prev, _ = state
        out, h1 = self.gru(x.unsqueeze(1), h0)
        dyn = out[:, 0]
        f = torch.cat([dyn, x, x - prev], dim=-1)
        residual = self.fusion(f)
        gate = torch.sigmoid(self.gate(f))
        y = F.normalize(x.new_zeros(x.shape[0], self.out_dim) +
                        x[:, :self.out_dim] + gate * residual, dim=-1)
        return y, (h1, x, y)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        b, t, _ = x.shape
        if mask is None:
            mask = torch.ones(b, t, dtype=torch.bool, device=x.device)
        else:
            mask = mask.bool()
        h, prev, last = self.new_state(b, x.device)
        seq = []
        for i in range(t):
            y, (h1, p1, l1) = self.step(x[:, i], (h, prev, last))
            valid = mask[:, i]
            h = torch.where(valid.view(1, b, 1), h1, h)
            prev = torch.where(valid.view(b, 1), p1, prev)
            last = torch.where(valid.view(b, 1), y, last)
            seq.append(last)
        seq = torch.stack(seq, dim=1)
        lengths = torch.clamp(mask.long().sum(1) - 1, min=0)
        final = seq[torch.arange(b, device=x.device), lengths]
        return F.normalize(final, dim=-1), F.normalize(seq, dim=-1)
