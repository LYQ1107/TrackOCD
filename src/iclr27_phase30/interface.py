"""Small support/query set correspondence interface for Phase30.

The module deliberately differs from the earlier GRU and residual adapters:
it uses a feed-forward causal track summary and an explicit set competition,
NULL and uncertainty heads.  No category, video, physical/semantic ID or
future value is consumed by ``forward``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class SupportSetCorrespondenceEncoder(nn.Module):
    def __init__(self, input_dim: int = 768, embedding_dim: int = 256, hidden_dim: int = 384):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.track = nn.Sequential(
            nn.LayerNorm(input_dim * 3),
            nn.Linear(input_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.quality = nn.Linear(embedding_dim, 1)
        self.null_head = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 1))
        self.uncertainty_head = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 1))

    def encode_track(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode one or many strictly causal track prefixes.

        ``seq`` is ``[..., T, D]`` and ``mask`` is ``[..., T]``.  Mean, max and
        last observations are causal summaries; no recurrent state is kept.
        """
        x = seq.float(); m = mask.bool()
        while m.ndim < x.ndim:
            m = m.unsqueeze(-1)
        count = m[..., 0].long().sum(-1, keepdim=True).clamp(min=1)
        mean = (x * m.float()).sum(-2) / count.float()
        neg = torch.finfo(x.dtype).min
        maxv = x.masked_fill(~m, neg).amax(-2)
        # The final valid row is obtained from the causal mask count.
        idx = (count.squeeze(-1) - 1).clamp(min=0)
        flat_x = x.reshape(-1, x.shape[-2], x.shape[-1])
        flat_idx = idx.reshape(-1)
        last = flat_x[torch.arange(flat_x.shape[0], device=x.device), flat_idx].reshape(*x.shape[:-2], x.shape[-1])
        stats = torch.cat([mean, maxv, last], dim=-1)
        return F.normalize(self.track(stats), dim=-1)

    def forward(
        self,
        query_seq: torch.Tensor,
        query_mask: torch.Tensor,
        support_seq: torch.Tensor,
        support_mask: torch.Tensor,
        support_set_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score a query against a variable-size support set.

        Shapes: query ``[B,T,D]``; support ``[B,S,T,D]``; temporal masks
        ``[B,T]``/``[B,S,T]``; support-set mask ``[B,S]``.
        """
        q = self.encode_track(query_seq, query_mask)
        b, s, t, d = support_seq.shape
        s_emb = self.encode_track(support_seq.reshape(b * s, t, d), support_mask.reshape(b * s, t)).reshape(b, s, -1)
        pair_scores = torch.sum(q[:, None, :] * s_emb, dim=-1)
        valid = support_set_mask.bool()
        masked_scores = pair_scores.masked_fill(~valid, -1e4)
        quality = self.quality(s_emb).squeeze(-1).masked_fill(~valid, -1e4)
        weights = torch.softmax(quality, dim=-1).masked_fill(~valid, 0.0)
        support_context = F.normalize(torch.sum(weights[..., None] * s_emb, dim=1), dim=-1)
        set_score = torch.sum(q * support_context, dim=-1)
        max_score = masked_scores.max(dim=-1).values
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1)
        null_features = torch.stack([max_score, entropy], dim=-1)
        null_logit = self.null_head(null_features).squeeze(-1)
        uncertainty = torch.sigmoid(self.uncertainty_head(null_features).squeeze(-1))
        return {"query_embedding": q, "support_embeddings": s_emb, "pair_scores": pair_scores, "set_score": set_score, "null_logit": null_logit, "uncertainty": uncertainty, "support_weights": weights}


def metadata(model: SupportSetCorrespondenceEncoder) -> dict[str, Any]:
    return {
        "architecture": "feed-forward causal track summary (mean/max/last) + support-set quality competition + NULL/uncertainty heads",
        "input_dim": model.input_dim,
        "embedding_dim": model.embedding_dim,
        "forbidden_inputs": ["category_id", "video_id", "physical_id", "semantic_id", "category_text", "future_frame", "future_track", "gt_bbox", "held_event_outcome", "StateMemory", "controller_action"],
    }
