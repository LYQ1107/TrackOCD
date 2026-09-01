"""Compact observability-gated semantic representation used by T0 and M1."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ObservabilitySemanticModel(nn.Module):
    def __init__(self, feature_dim: int, geometry_dim: int, embedding_dim: int,
                 num_known: int, num_views: int = 4):
        super().__init__()
        h = embedding_dim
        self.feature_dim = feature_dim
        self.geometry_dim = geometry_dim
        self.embedding_dim = embedding_dim
        self.num_known = num_known
        self.num_views = num_views
        self.view_projection = nn.Sequential(nn.Linear(feature_dim, h), nn.LayerNorm(h), nn.GELU())
        self.view_attention = nn.Sequential(nn.Linear(h + geometry_dim, h // 2), nn.GELU(), nn.Linear(h // 2, 1))
        self.previous_projection = nn.Sequential(nn.Linear(feature_dim, h), nn.LayerNorm(h), nn.GELU())
        self.temporal_gate = nn.Sequential(nn.Linear(h * 2 + geometry_dim, h), nn.GELU(), nn.Linear(h, h), nn.Sigmoid())
        self.semantic_projection = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h))
        self.observability_head = nn.Sequential(nn.Linear(h + geometry_dim, h // 2), nn.GELU(), nn.Linear(h // 2, 1))
        self.known_rejector = nn.Sequential(nn.Linear(h + geometry_dim, h // 2), nn.GELU(), nn.Linear(h // 2, 1))
        self.known_weight = nn.Parameter(torch.empty(num_known, h))
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.pair_head = nn.Sequential(nn.Linear(h * 2 + 1, h), nn.GELU(), nn.Dropout(.10), nn.Linear(h, 1))
        nn.init.normal_(self.known_weight, std=.02)

    def project_base(self, feature: torch.Tensor) -> torch.Tensor:
        h = self.view_projection(feature)
        return F.normalize(self.semantic_projection(h), dim=-1)

    def forward(self, views: torch.Tensor, previous_raw: torch.Tensor,
                geometry: torch.Tensor) -> dict[str, torch.Tensor]:
        # views: [B,V,D]. Physical ID and provenance never enter this tensor.
        b, v, _ = views.shape
        vh = self.view_projection(views)
        g = geometry[:, None, :].expand(b, v, geometry.shape[-1])
        weights = torch.softmax(self.view_attention(torch.cat([vh, g], dim=-1)).squeeze(-1), dim=1)
        current = (vh * weights[:, :, None]).sum(dim=1)
        previous = self.previous_projection(previous_raw)
        gate = self.temporal_gate(torch.cat([current, previous, geometry], dim=-1))
        causal = gate * current + (1.0 - gate) * previous
        semantic = F.normalize(self.semantic_projection(causal), dim=-1)
        head_input = torch.cat([semantic, geometry], dim=-1)
        class_weight = F.normalize(self.known_weight, dim=-1)
        class_logits = F.linear(semantic, class_weight) * self.logit_scale.exp().clamp(max=100.0)
        return {
            "semantic": semantic,
            "observability_logit": self.observability_head(head_input).squeeze(-1),
            "known_logit": self.known_rejector(head_input).squeeze(-1),
            "class_logits": class_logits,
            "attention": weights,
            "temporal_gate": gate
        }

    def teacher(self, gt_views: torch.Tensor) -> torch.Tensor:
        h = self.view_projection(gt_views).mean(dim=1)
        return F.normalize(self.semantic_projection(h), dim=-1)

    def pair_logits(self, query: torch.Tensor, state: torch.Tensor,
                    state_count_log: torch.Tensor) -> torch.Tensor:
        x = torch.cat([torch.abs(query - state), query * state, state_count_log[:, None]], dim=-1)
        return self.pair_head(x).squeeze(-1)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
