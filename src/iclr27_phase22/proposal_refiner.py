"""Small, class-agnostic causal proposal box/quality refiner.

The model consumes only frozen current-row visual features and causal
geometry/score/history fields.  Category IDs, physical IDs, semantic IDs,
GT boxes, row IoU, and future rows are deliberately absent from the input.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


GEOM_FIELDS = (
    "score", "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm",
    "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log",
    "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm",
    "causal_prefix_age_norm", "causal_box_stability_iou",
)


class ProposalRefiner(nn.Module):
    """A single shared representation with box-delta and quality heads."""

    def __init__(self, visual_dim: int = 1536, geom_dim: int = len(GEOM_FIELDS), hidden: int = 256) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.geom_dim = int(geom_dim)
        self.visual = nn.Sequential(nn.LayerNorm(self.visual_dim), nn.Linear(self.visual_dim, hidden), nn.GELU())
        self.geometry = nn.Sequential(nn.LayerNorm(self.geom_dim), nn.Linear(self.geom_dim, 64), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(hidden + 64, hidden), nn.GELU(), nn.Dropout(p=0.05))
        self.box_delta = nn.Linear(hidden, 4)
        self.quality = nn.Linear(hidden, 1)
        # Refinement is a residual over the frozen proposal.  Starting from
        # the identity is both safer for the causal stream and an explicit
        # guard against an untrained head destroying a usable box.
        nn.init.zeros_(self.box_delta.weight)
        nn.init.zeros_(self.box_delta.bias)

    def forward(self, visual: torch.Tensor, geom: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.fuse(torch.cat([self.visual(visual), self.geometry(geom)], dim=-1))
        # A bounded delta keeps a refinement from creating arbitrary boxes.
        return {"box_delta": 0.75 * torch.tanh(self.box_delta(h)), "quality_logit": self.quality(h).squeeze(-1)}


def corrected_box(box_xyxy_norm: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply and clamp normalized xyxy deltas without using any future data."""
    return torch.clamp(box_xyxy_norm + delta, 0.0, 1.0)


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)
    iw = torch.clamp(torch.minimum(ax2, bx2) - torch.maximum(ax1, bx1), min=0.0)
    ih = torch.clamp(torch.minimum(ay2, by2) - torch.maximum(ay1, by1), min=0.0)
    inter = iw * ih
    aa = torch.clamp(ax2 - ax1, min=0.0) * torch.clamp(ay2 - ay1, min=0.0)
    ab = torch.clamp(bx2 - bx1, min=0.0) * torch.clamp(by2 - by1, min=0.0)
    return inter / torch.clamp(aa + ab - inter, min=1e-8)


def state_dict_metadata(model: ProposalRefiner) -> dict[str, Any]:
    return {"visual_dim": model.visual_dim, "geom_dim": model.geom_dim, "hidden": model.visual[1].out_features,
            "input_fields": ["frozen_dinov2_cls", "frozen_dinov2_roi", *GEOM_FIELDS],
            "forbidden_input_fields": ["gt_bbox_xyxy", "row_iou", "gt_category_id_common", "physical_id", "semantic_id", "future_frame"],
            "outputs": ["box_delta_normalized_xyxy", "quality_logit"]}
