from __future__ import annotations
import torch
from torch import nn
class PolicyDistilledGate(nn.Module):
    def __init__(self, init_logit: float=1.5):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(5),nn.Linear(5,16),nn.GELU(),nn.Linear(16,1)); nn.init.zeros_(self.net[-1].weight); nn.init.constant_(self.net[-1].bias,init_logit)
    def forward(self, raw_margin, bridge_margin, support_quality, alpha, uncertainty):
        x=torch.stack([raw_margin,bridge_margin,support_quality,alpha,uncertainty],-1); return torch.sigmoid(self.net(x).squeeze(-1))
