from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
class SafetyVectorBridge(nn.Module):
    def __init__(self, alpha_max: float=0.15):
        super().__init__(); self.alpha_max=float(alpha_max)
        self.gate=nn.Sequential(nn.LayerNorm(5),nn.Linear(5,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)
    def forward(self, raw_vec, support_context, raw_score, support_quality, valid=True):
        if support_context is None: return F.normalize(raw_vec,dim=-1), torch.zeros(raw_vec.shape[:-1],device=raw_vec.device), torch.zeros(raw_vec.shape[:-1],device=raw_vec.device)
        diff=support_context-raw_vec; feat=torch.stack([raw_score,support_quality,(raw_vec*support_context).sum(-1),diff.abs().mean(-1),raw_vec.mean(-1)],-1)
        # sigmoid(g)*g is zero at initialization with non-zero gradient; clamp
        # enforces the registered [0, alpha_max] safety bound.
        g=self.gate(feat).squeeze(-1)
        alpha=torch.clamp(self.alpha_max*torch.sigmoid(g)*g, min=0.0, max=self.alpha_max)
        if not valid: alpha=torch.zeros_like(alpha)
        z=F.normalize(raw_vec+alpha.unsqueeze(-1)*torch.clamp(diff,-0.25,0.25),dim=-1)
        return z,alpha,diff
