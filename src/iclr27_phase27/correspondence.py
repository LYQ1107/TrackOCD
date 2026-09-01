from __future__ import annotations
from typing import Any
import torch
import torch.nn.functional as F
from torch import nn

class TrackCorrespondenceEncoder(nn.Module):
    """One causal GRU; no action head, memory or category feature."""
    def __init__(self, input_dim: int = 768, hidden: int = 128, output_dim: int = 768):
        super().__init__(); self.input_dim=input_dim; self.hidden=hidden; self.output_dim=output_dim
        self.norm=nn.LayerNorm(input_dim); self.gru=nn.GRU(input_dim,hidden,batch_first=True); self.proj=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,output_dim))
    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h,_=self.gru(self.norm(seq.float())); n=mask.long().sum(1).clamp(min=1); last=h[torch.arange(h.shape[0],device=h.device),n-1]; return F.normalize(self.proj(last),dim=-1)
def metadata(m: TrackCorrespondenceEncoder) -> dict[str, Any]:
    return {"architecture":"one-layer causal GRU over fused DINOv2 CLS/ROI", "input_dim":m.input_dim,"hidden":m.hidden,"output_dim":m.output_dim,"forbidden_inputs":["gt_bbox","category_id_feature","physical_id","semantic_id","category_text","future_frame","future_track","StateMemory","controller_action"]}
