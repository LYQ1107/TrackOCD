"""Small anchored candidate-level relation scorer.

It never emits a new object embedding.  The only learned outputs are a scalar
local relation delta and confidence, combined with a supplied raw cosine.
"""
from __future__ import annotations

import torch
from torch import nn


class AnchoredRelationReranker(nn.Module):
    pair_dim = 1536
    summary_dim = 13

    def __init__(self) -> None:
        super().__init__()
        self.pair_token = nn.Sequential(
            nn.Linear(self.pair_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.quality = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
        self.delta_head = nn.Sequential(nn.Linear(128 + self.summary_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.conf_head = nn.Sequential(nn.Linear(128 + self.summary_dim, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.delta_head[-1].weight); nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.conf_head[-1].weight); nn.init.zeros_(self.conf_head[-1].bias)

    def forward_features(
        self,
        pair_tokens: torch.Tensor,
        summary: torch.Tensor,
        raw_cosine: torch.Tensor | float,
    ) -> dict[str, torch.Tensor]:
        if pair_tokens.ndim != 2 or pair_tokens.shape[-1] != self.pair_dim:
            raise ValueError(f"pair_tokens must be [N,1536], got {tuple(pair_tokens.shape)}")
        if summary.shape[-1] != self.summary_dim:
            raise ValueError(f"summary must end in 13, got {tuple(summary.shape)}")
        token = self.pair_token(pair_tokens)
        # Five causal quality inputs are a fixed subset of the 13 summary
        # fields: raw/matched central tendency, coverage, and query stability.
        quality_input = torch.stack([summary[..., 0], summary[..., 1], summary[..., 2], summary[..., 8], summary[..., 11]], dim=-1)
        weights = self.quality(quality_input).squeeze(-1)
        pooled = (weights.unsqueeze(-1) * token).sum(dim=-2) / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0)
        if summary.ndim == 1:
            summary = summary.unsqueeze(0)
        context = torch.cat([pooled, summary], dim=-1)
        delta = self.delta_head(context).squeeze(-1)
        confidence = torch.sigmoid(self.conf_head(context).squeeze(-1))
        raw = torch.as_tensor(raw_cosine, dtype=context.dtype, device=context.device)
        final = raw + confidence * delta
        return {"delta": delta, "confidence": confidence, "final": final, "weights": weights, "pooled": pooled}

    def forward(self, pair_tokens: torch.Tensor, summary: torch.Tensor, raw_cosine: torch.Tensor | float) -> dict[str, torch.Tensor]:
        return self.forward_features(pair_tokens, summary, raw_cosine)

