"""Small set-aware class-agnostic candidate selector for Phase24."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SetAwareCandidateSelector(nn.Module):
    """Score a complete causal candidate set jointly.

    Candidate features are key-aligned DINOv2 CLS/ROI plus geometry and
    transform metadata.  Masked mean/max context lets the score depend on the
    set without carrying an identity, category, GT or online-memory signal.
    """

    def __init__(self, visual_dim: int = 1536, geom_dim: int = 22, hidden: int = 128, context: int = 64) -> None:
        super().__init__()
        self.visual_dim, self.geom_dim, self.hidden, self.context_dim = visual_dim, geom_dim, hidden, context
        self.visual = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, 96), nn.GELU())
        self.geometry = nn.Sequential(nn.LayerNorm(geom_dim), nn.Linear(geom_dim, 64), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(160, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.context = nn.Sequential(nn.Linear(hidden * 2, context), nn.GELU())
        self.quality = nn.Sequential(nn.Linear(hidden + context, 64), nn.GELU(), nn.Linear(64, 1))
        self.uncertainty = nn.Sequential(nn.Linear(hidden + context, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, visual: torch.Tensor, geometry: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # visual/geometry: [B,N,D], mask: [B,N] bool/float
        x = self.candidate(torch.cat([self.visual(visual), self.geometry(geometry)], dim=-1))
        m = mask.float().unsqueeze(-1)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.)
        mean = (x * m).sum(dim=1, keepdim=True) / denom
        neg = torch.where(m > 0, x, torch.full_like(x, -1e4))
        mx = neg.max(dim=1, keepdim=True).values
        ctx = self.context(torch.cat([mean, mx], dim=-1)).expand(-1, x.shape[1], -1)
        z = torch.cat([x, ctx], dim=-1)
        return self.quality(z).squeeze(-1), self.uncertainty(z).squeeze(-1)


def metadata(model: SetAwareCandidateSelector) -> dict[str, Any]:
    return {
        "visual_dim": model.visual_dim, "geometry_dim": model.geom_dim,
        "hidden": model.hidden, "context_dim": model.context_dim,
        "inputs": ["aligned_dinov2_cls", "aligned_dinov2_roi", "parent_geometry",
                    "candidate_box_xyxy", "raw_score", "transform_metadata",
                    "causal_age", "history_stability", "parent_frame"],
        "outputs": ["candidate_quality_logit", "candidate_uncertainty_logit"],
        "forbidden": ["gt_bbox_xyxy", "row_iou", "gt_category_id_common",
                       "physical_id", "semantic_id", "future_frame", "StateMemory"],
    }

