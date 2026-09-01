"""Small class-agnostic causal proposal-source head."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.iclr27_phase26.protocol import SOURCE_ANCHORS


class ProposalSourceGenerator(nn.Module):
    """Predict several bounded candidate boxes from one causal proposal row.

    The head adds candidates; it never changes a physical track or parent row.
    All output candidates retain the parent's assigned bit at evaluation time.
    """
    def __init__(self, visual_dim: int = 1536, geom_dim: int = 15,
                 hidden: int = 128, num_sources: int = 8) -> None:
        super().__init__()
        self.visual_dim, self.geom_dim, self.hidden, self.num_sources = visual_dim, geom_dim, hidden, num_sources
        self.visual = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, 96), nn.GELU())
        self.geometry = nn.Sequential(nn.LayerNorm(geom_dim), nn.Linear(geom_dim, 48), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(144, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.delta = nn.Linear(hidden, num_sources * 4)
        self.quality = nn.Linear(hidden, num_sources)
        self.register_buffer("anchors", torch.as_tensor(SOURCE_ANCHORS[:num_sources], dtype=torch.float32), persistent=False)
        # Start as a bounded, low-motion source extension; training can move it.
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.quality.weight); nn.init.constant_(self.quality.bias, -1.0)

    def forward(self, visual: torch.Tensor, geometry: torch.Tensor,
                base_box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fusion(torch.cat([self.visual(visual), self.geometry(geometry)], dim=-1))
        d = torch.tanh(self.delta(h).view(-1, self.num_sources, 4))
        # Fixed causal anchor grid gives a genuine multi-candidate source
        # branch; the learned residual is bounded by the parent geometry.
        cx = (base_box[:, 0] + base_box[:, 2]) * .5; cy = (base_box[:, 1] + base_box[:, 3]) * .5
        bw = torch.clamp(base_box[:, 2] - base_box[:, 0], min=1e-4); bh = torch.clamp(base_box[:, 3] - base_box[:, 1], min=1e-4)
        a = self.anchors.to(base_box).unsqueeze(0); acx = cx.unsqueeze(1) + a[..., 1] * bw.unsqueeze(1); acy = cy.unsqueeze(1) + a[..., 2] * bh.unsqueeze(1); aw = bw.unsqueeze(1) * a[..., 0]; ah = bh.unsqueeze(1) * a[..., 0]
        anchor_boxes = torch.stack([acx - .5*aw, acy - .5*ah, acx + .5*aw, acy + .5*ah], dim=-1)
        boxes = torch.clamp(anchor_boxes + .25 * d * torch.stack([bw, bh, bw, bh], dim=-1).unsqueeze(1), 0.0, 1.0)
        quality = self.quality(h)
        return boxes, quality


def metadata(model: ProposalSourceGenerator) -> dict[str, Any]:
    return {
        "architecture": "LayerNorm/linear CLS+ROI, geometry projection, two-layer fusion, eight bounded xyxy source candidates plus quality logits",
        "visual_dim": model.visual_dim, "geom_dim": model.geom_dim,
        "hidden": model.hidden, "num_sources": model.num_sources,
        "anchor_grid": SOURCE_ANCHORS.tolist(),
        "inputs": ["key_aligned_dinov2_cls", "key_aligned_dinov2_roi", "causal_bbox_geometry", "raw_score", "causal_age", "history_stability"],
        "outputs": ["eight_candidate_box_coordinates", "candidate_quality_logits"],
        "forbidden_inputs": ["gt_bbox_xyxy", "row_iou", "category_id", "physical_id", "semantic_id", "text", "future_frame", "future_track", "StateMemory"],
    }
