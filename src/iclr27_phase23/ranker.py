"""Small class-agnostic quality scorer for the fixed candidate pool."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CandidateQualityRanker(nn.Module):
    """Score each causal candidate independently; no state or IDs are used."""

    def __init__(self, visual_dim: int = 1536, geom_dim: int = 22, hidden: int = 192) -> None:
        super().__init__()
        self.visual_dim, self.geom_dim, self.hidden = int(visual_dim), int(geom_dim), int(hidden)
        self.visual = nn.Sequential(nn.LayerNorm(self.visual_dim), nn.Linear(self.visual_dim, 128), nn.GELU())
        self.geometry = nn.Sequential(nn.LayerNorm(self.geom_dim), nn.Linear(self.geom_dim, 64), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(192, self.hidden), nn.GELU(), nn.Dropout(0.05), nn.Linear(self.hidden, 1))

    def forward(self, visual: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.visual(visual), self.geometry(geometry)], dim=-1)).squeeze(-1)


def metadata(model: CandidateQualityRanker) -> dict[str, Any]:
    return {"visual_dim": model.visual_dim, "geometry_dim": model.geom_dim, "hidden": model.hidden,
            "inputs": ["aligned_dinov2_cls", "aligned_dinov2_roi", "candidate_box_xyxy", "parent_box_geometry", "raw_score", "causal_age", "history_stability", "transform_id"],
            "forbidden": ["gt_bbox_xyxy", "row_iou", "gt_category_id_common", "physical_id", "semantic_id", "future_frame"],
            "output": "candidate_quality_logit"}
