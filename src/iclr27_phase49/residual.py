from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

class RawPreservingResidualBridge(nn.Module):
    """768-D raw anchor with a bounded, support-conditioned residual."""
    def __init__(self, dim: int = 768, alpha_max: float = 0.05):
        super().__init__(); self.dim=int(dim); self.alpha_max=float(alpha_max)
        self.residual = nn.Sequential(nn.LayerNorm(dim*3), nn.Linear(dim*3,128), nn.GELU(), nn.Linear(128,dim))
        self.alpha_head = nn.Sequential(nn.LayerNorm(5), nn.Linear(5,32), nn.GELU(), nn.Linear(32,1))
        # Keep alpha at zero initially (exact raw output) but leave a tiny
        # residual gradient so the alpha gate can learn from TRAIN examples.
        nn.init.normal_(self.residual[-1].weight, mean=0.0, std=1e-3); nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.alpha_head[-1].weight); nn.init.zeros_(self.alpha_head[-1].bias)
    def forward(self, raw: torch.Tensor, support: torch.Tensor|None = None, support_mask: torch.Tensor|None = None, valid_support: bool=True):
        raw = F.normalize(raw.float(), dim=-1)
        if support is None or support.numel()==0 or not valid_support:
            return raw, torch.zeros(raw.shape[:-1],device=raw.device), torch.zeros_like(raw)
        s = F.normalize(support.float(), dim=-1)
        if s.ndim == raw.ndim: s=s.unsqueeze(0)
        if support_mask is None: support_mask=torch.ones(s.shape[:-1],device=s.device,dtype=torch.bool)
        m=support_mask.bool().unsqueeze(-1)
        ctx=(s*m.float()).sum(-2)/m.sum(-2).clamp_min(1)
        ctx=F.normalize(ctx,dim=-1)
        diff=ctx-raw
        feat=torch.stack([(raw*ctx).sum(-1), diff.abs().mean(-1), ctx.mean(-1), raw.mean(-1), m.squeeze(-1).sum(-1).float()],-1)
        g=self.alpha_head(feat).squeeze(-1)
        # Bounded and differentiable at the zero-initialized anchor; this keeps
        # exact raw fallback for invalid support while allowing TRAIN gradients.
        alpha=self.alpha_max*torch.tanh(g)
        r=torch.tanh(self.residual(torch.cat([raw,ctx,diff],-1)))
        z=F.normalize(raw+alpha.unsqueeze(-1)*r,dim=-1)
        return z,alpha,r
