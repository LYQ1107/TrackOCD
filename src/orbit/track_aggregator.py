"""Geometry-preserving reliability-aware track aggregation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryPreservingAdapter(nn.Module):
    """Low-capacity residual adapter; identity-like initialization."""

    def __init__(self, dim: int = 768, bottleneck: int = 128):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.net(x), dim=-1)


class ReliabilityAggregator(nn.Module):
    """Attention pooling over adapted frame features with geometry loss."""

    def __init__(self, dim: int = 768, bottleneck: int = 128, use_reliability: bool = True):
        super().__init__()
        self.adapter = GeometryPreservingAdapter(dim, bottleneck)
        self.use_reliability = use_reliability
        self.attn = nn.Linear(dim + 3, 1)
        self.dim = dim

    def aggregate(self, x: torch.Tensor, mask: torch.Tensor):
        """x: B,T,d (already L2-normalized frame embeddings).
        mask: B,T bool (True = valid)."""
        B, T, d = x.shape
        y = self.adapter(x)
        c0 = F.normalize(y.detach().mean(dim=1), dim=-1)  # B,d
        cos = (y * c0.unsqueeze(1)).sum(-1)  # B,T
        tpos = torch.linspace(0.0, 1.0, T, device=x.device).unsqueeze(0).expand(B, T)
        delta = 1.0 - cos
        length = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # B,1
        if self.use_reliability:
            feat = torch.cat([y, cos.unsqueeze(-1), delta.unsqueeze(-1), tpos.unsqueeze(-1)], dim=-1)
            logits = self.attn(feat).squeeze(-1)  # B,T
            logits = logits.masked_fill(~mask, float("-inf"))
            w = torch.softmax(logits, dim=1)
        else:
            w = mask.float() / length
        z = F.normalize((w.unsqueeze(-1) * y).sum(dim=1), dim=-1)
        z0 = F.normalize(x.mean(dim=1), dim=-1)
        return {
            "z": z,
            "z0": z0.detach(),
            "weights": w,
            "y": y,
            "cos": cos,
            "length": length.squeeze(-1),
        }


def geometry_loss(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Preserve pairwise cosine structure of original DINO track means."""
    if z.shape[0] < 2:
        return torch.zeros((), device=z.device)
    cz = torch.mm(z, z.t())
    cz0 = torch.mm(z0, z0.t())
    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    return torch.abs(cz[mask] - cz0[mask]).mean()
