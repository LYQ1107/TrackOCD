from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class DomainAlignedResidualEncoder(nn.Module):
    """A conservative, non-recurrent residual metric adapter.

    The zero-initialized residual makes the initial output exactly the causal
    mean feature.  Video/category metadata never enters ``forward``.
    """

    def __init__(self, input_dim: int = 768, output_dim: int = 768, residual_scale: float = 0.10):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.residual_scale = float(residual_scale)
        self.norm = nn.LayerNorm(input_dim * 3)
        self.residual = nn.Linear(input_dim * 3, output_dim, bias=False)
        nn.init.zeros_(self.residual.weight)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = seq.float()
        m = mask.bool()
        count = m.long().sum(1, keepdim=True).clamp(min=1)
        mf = m.unsqueeze(-1).float()
        mean = (x * mf).sum(1) / count.float()
        idx = count.squeeze(1) - 1
        last = x[torch.arange(x.shape[0], device=x.device), idx]
        stats = torch.cat([mean, last, (last - mean).abs()], dim=-1)
        delta = torch.tanh(self.residual(self.norm(stats))) * self.residual_scale
        return F.normalize(mean + delta, dim=-1)


def metadata(model: DomainAlignedResidualEncoder) -> dict[str, Any]:
    return {
        "architecture": "zero-initialized residual metric adapter over causal mean/last/abs-delta",
        "input_dim": model.input_dim,
        "output_dim": model.output_dim,
        "residual_scale": model.residual_scale,
        "forbidden_inputs": [
            "category_id_feature", "physical_id", "semantic_id", "category_text",
            "future_frame", "future_track", "gt_bbox", "held_gt", "StateMemory",
            "controller_action", "video_id_feature",
        ],
    }
