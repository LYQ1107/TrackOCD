from __future__ import annotations
import torch
from torch import nn

class PhysicalUnionGate(nn.Module):
    def __init__(self, in_dim:int=10, hidden:int=64):
        super().__init__(); self.net=nn.Sequential(nn.Linear(in_dim,hidden),nn.LayerNorm(hidden),nn.Tanh(),nn.Linear(hidden,1))
    def forward(self,x): return self.net(x).squeeze(-1)
