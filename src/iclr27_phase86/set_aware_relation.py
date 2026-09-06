"""Registered Phase86 U2 set-aware raw-preserving relation reranker."""
from __future__ import annotations
import torch
from torch import nn

class SetAwareResidualReranker(nn.Module):
    def __init__(self, candidate_dim: int = 19, context_dim: int = 10):
        super().__init__()
        self.candidate = nn.Sequential(nn.Linear(candidate_dim + context_dim, 128), nn.LayerNorm(128), nn.GELU())
        self.context = nn.Linear(context_dim, 128)
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True, norm_first=True, dropout=0.1, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(128, 1)
    def forward(self, candidates: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 2: raise ValueError('candidates must be [N,D]')
        c = context if context.ndim == 1 else context[0]
        c = c.reshape(1, -1).expand(candidates.shape[0], -1)
        tokens = self.candidate(torch.cat([candidates, c], dim=-1))
        token = self.context(context.reshape(1, -1))
        encoded = self.encoder(torch.cat([token, tokens], dim=0).unsqueeze(0)).squeeze(0)[1:]
        return 0.05 * torch.tanh(self.head(encoded).squeeze(-1))
