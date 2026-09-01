"""Single, frozen-track temporal correspondence encoder for Phase26."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class TrackCorrespondenceEncoder(nn.Module):
    def __init__(self, input_dim: int = 768, hidden: int = 128, output_dim: int = 768) -> None:
        super().__init__(); self.input_dim, self.hidden, self.output_dim = input_dim, hidden, output_dim
        self.input_norm = nn.LayerNorm(input_dim); self.gru = nn.GRU(input_dim, hidden, num_layers=1, batch_first=True); self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, output_dim))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_norm(sequence.float())
        out, _ = self.gru(x)
        if mask is None: h = out[:, -1]
        else:
            lengths = mask.long().sum(dim=1).clamp(min=1); h = out[torch.arange(out.shape[0], device=out.device), lengths - 1]
        return F.normalize(self.proj(h), dim=-1)


def metadata(model: TrackCorrespondenceEncoder) -> dict[str, Any]:
    return {"architecture": "one-layer causal GRU over fused DINOv2 CLS/ROI row features", "input_dim": model.input_dim, "hidden": model.hidden, "output_dim": model.output_dim, "inputs": ["key_aligned_dinov2_cls_roi", "causal_track_prefix"], "forbidden_inputs": ["gt_bbox", "category_text", "category_id_feature", "physical_id", "semantic_id", "future_frames", "StateMemory", "controller_action"]}
