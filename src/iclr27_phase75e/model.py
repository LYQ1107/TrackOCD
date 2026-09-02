"""The single registered Phase75E learnable component.

This is deliberately a feature-space, LoRA-inspired residual.  It does not
modify DINO attention weights and has no category, identity, text, or state
memory inputs.  ``B`` is zero initialized, so a freshly constructed adapter
is an exact (up to the input normalization) raw-feature mapping.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LowRankFeatureAdapter(nn.Module):
    """Rank-8 bounded residual adapter for 768-D frame features."""

    def __init__(self, dim: int = 768, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if dim <= 0 or rank <= 0:
            raise ValueError("dim and rank must be positive")
        self.dim = int(dim)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.A = nn.Linear(self.dim, self.rank, bias=False)
        self.B = nn.Linear(self.rank, self.dim, bias=False)
        # Standard small initialization for the down projection and the
        # standard residual-anchor zero initialization for the up projection.
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {tuple(x.shape)}")
        raw = F.normalize(x.float(), dim=-1)
        delta = self.B(self.A(raw))
        return F.normalize(raw + self.scale * delta, dim=-1)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """Return the unscaled residual (useful for drift diagnostics)."""
        raw = F.normalize(x.float(), dim=-1)
        return self.B(self.A(raw))
