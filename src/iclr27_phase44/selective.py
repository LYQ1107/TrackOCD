from __future__ import annotations
import torch
from torch import nn
class CalibratedGate(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(5),nn.Linear(5,16),nn.GELU(),nn.Linear(16,1)); nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,*x): return torch.sigmoid(self.net(torch.stack(x,-1)).squeeze(-1))
