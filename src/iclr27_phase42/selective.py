from __future__ import annotations
import torch
from torch import nn
class SelectiveGate(nn.Module):
    def __init__(self):
        super().__init__(); self.logit=nn.Sequential(nn.LayerNorm(5),nn.Linear(5,16),nn.GELU(),nn.Linear(16,1)); nn.init.zeros_(self.logit[-1].weight); nn.init.constant_(self.logit[-1].bias,-2.0)
    def forward(self, raw_margin, bridge_margin, support_quality, alpha, uncertainty):
        x=torch.stack([raw_margin,bridge_margin,support_quality,alpha,uncertainty],-1); return torch.sigmoid(self.logit(x).squeeze(-1))
