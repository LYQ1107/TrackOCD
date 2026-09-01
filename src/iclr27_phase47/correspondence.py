from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

class DomainAlignedEncoder(nn.Module):
    """Small class-agnostic encoder; metadata never enters the network."""
    def __init__(self, input_dim: int = 768, hidden: int = 256, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.float()), dim=-1)
