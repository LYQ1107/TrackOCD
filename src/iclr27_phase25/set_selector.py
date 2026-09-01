"""Phase25 set-aware selector: one self-attention block over candidates."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ProposalSetAttentionSelector(nn.Module):
    """Class-agnostic quality/uncertainty model over a causal candidate set.

    The model is intentionally different from Phase24's masked mean/max MLP:
    candidate tokens exchange information through a single two-head attention
    block.  No category, identity, GT or online-memory field is accepted.
    """

    def __init__(self, visual_dim: int = 1536, geom_dim: int = 22, hidden: int = 128, heads: int = 2) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.visual_dim, self.geom_dim, self.hidden, self.heads = visual_dim, geom_dim, hidden, heads
        self.visual = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, 80), nn.GELU())
        self.geometry = nn.Sequential(nn.LayerNorm(geom_dim), nn.Linear(geom_dim, 48), nn.GELU())
        self.token = nn.Sequential(nn.Linear(128, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.quality = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))
        self.uncertainty = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, visual: torch.Tensor, geometry: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token(torch.cat([self.visual(visual), self.geometry(geometry)], dim=-1))
        key_padding = ~mask.bool()
        attn, _ = self.attn(x, x, x, key_padding_mask=key_padding, need_weights=False)
        x = self.norm(x + attn)
        x = self.norm(x + self.ff(x))
        q = self.quality(x).squeeze(-1)
        u = self.uncertainty(x).squeeze(-1)
        return q, u


def metadata(model: ProposalSetAttentionSelector) -> dict[str, Any]:
    return {
        "architecture": "LayerNorm/linear CLS+ROI and geometry, one 2-head self-attention block, residual FFN, quality+uncertainty heads",
        "visual_dim": model.visual_dim, "geometry_dim": model.geom_dim, "hidden": model.hidden, "heads": model.heads,
        "inputs": ["key_aligned_dinov2_cls", "key_aligned_dinov2_roi", "candidate_box_geometry", "transform_metadata", "raw_score", "parent_frame", "causal_age", "history_stability"],
        "outputs": ["candidate_quality_logit", "candidate_uncertainty_logit"],
        "forbidden_inputs": ["gt_bbox_xyxy", "row_iou", "category_id", "physical_id", "semantic_id", "text", "future_frame", "future_track", "StateMemory"],
    }
