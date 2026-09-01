"""Small causal hybrid trajectory semantic encoder.

The encoder consumes frozen semantic foundation frame features (TSE output),
keeps a private GRU state for each physical track, and fuses the current
foundation feature with causal dynamics (GRU state and first difference).  It
does not receive the physical track ID as a feature and contains no semantic
memory or assign/create logic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridTrajectoryEncoder(nn.Module):
    def __init__(self, dim: int = 128, hidden: int = 128,
                 out_dim: int = 128):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)
        self.gru = nn.GRU(self.dim, self.hidden, batch_first=True)
        fusion_dim = self.hidden + self.dim + self.dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.out_dim),
        )
        self.gate = nn.Linear(fusion_dim, self.out_dim)
        # Start close to the frozen semantic feature so the prototype is a
        # representation reset, not an abrupt knownness/memory reset.
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
        """One causal step; ``x`` is (B, dim), state is prefix-only."""
        h0, prev, _ = state
        out, h1 = self.gru(x.unsqueeze(1), h0)
        dyn = out[:, 0]
        delta = x - prev
        f = torch.cat([dyn, x, delta], dim=-1)
        residual = self.fusion(f)
        gate = torch.sigmoid(self.gate(f))
        y = F.normalize(x + gate * residual, dim=-1)
        return y, (h1, x, y)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """Causal sequence encoding.

        ``x`` is (B,T,dim), mask is (B,T). Padded steps do not advance the
        hidden state. Returns the final valid embedding and all prefix outputs.
        """
        B, T, _ = x.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        else:
            mask = mask.bool()
        h, prev, last = self.new_state(B, x.device)
        outputs = []
        for t in range(T):
            y_new, (h_new, prev_new, _) = self.step(x[:, t], (h, prev, last))
            valid = mask[:, t]
            vh = valid.view(1, B, 1)
            vx = valid.view(B, 1)
            h = torch.where(vh, h_new, h)
            prev = torch.where(vx, x[:, t], prev)
            last = torch.where(vx, y_new, last)
            outputs.append(last)
        seq = torch.stack(outputs, dim=1)
        lengths = torch.clamp(mask.long().sum(dim=1) - 1, min=0)
        final = seq[torch.arange(B, device=x.device), lengths]
        return F.normalize(final, dim=-1), F.normalize(seq, dim=-1)
