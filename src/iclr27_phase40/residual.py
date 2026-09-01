from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class RawPreservingSupportResidual(nn.Module):
    """Bounded score residual; raw cosine is always the anchor."""
    def __init__(self, beta_max: float = 0.15, input_dim: int = 6):
        super().__init__(); self.beta_max=float(beta_max)
        self.mlp=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,32),nn.GELU(),nn.Linear(32,1))
        # Zero initialization makes the initial model exactly raw cosine.
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, raw, q, c, support_max, support_count, valid_support=True):
        # q/c are normalized embeddings. All quantities are causal and ID-free.
        feat=torch.stack([raw, support_max, raw-support_max, (q-c).abs().mean(-1), q.mean(-1), c.mean(-1)],dim=-1)
        delta=self.mlp(feat).squeeze(-1).tanh()
        beta=self.beta_max*torch.sigmoid(raw*4.0 + support_max*2.0)
        if not valid_support: beta=torch.zeros_like(beta)
        return raw + beta*delta, delta, beta

def pair_features(q: torch.Tensor, c: torch.Tensor, support: torch.Tensor|None):
    raw=(q*c).sum(-1)
    if support is None or support.numel()==0:
        sm=torch.zeros_like(raw); n=torch.zeros_like(raw)
    else:
        sm=(support@c.transpose(-1,-2)).max(dim=-2).values.squeeze(-1) if support.ndim==3 else (support*c).sum(-1).max(dim=-1).values
        n=torch.full_like(raw,float(support.shape[-2] if support.ndim==3 else support.shape[0]))
    return raw,sm,n
