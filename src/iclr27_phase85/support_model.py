"""Small raw-anchored set selector used by the registered Phase85 route."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

class SupportReranker(nn.Module):
    def __init__(self, candidate_dim: int=19, context_dim: int=10, hidden: int=64, residual_scale: float=0.05):
        super().__init__(); self.residual_scale=float(residual_scale)
        self.pair=nn.Sequential(nn.Linear(candidate_dim+context_dim,hidden),nn.LayerNorm(hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.defer=nn.Sequential(nn.Linear(context_dim,hidden),nn.LayerNorm(hidden),nn.Tanh(),nn.Linear(hidden,1))
    def forward(self, candidates: torch.Tensor, context: torch.Tensor):
        # candidates [N,D], context [C] or [N,C]; both paths are causal.
        base_context=context if context.ndim==1 else context[0]
        if context.ndim==1: context=context.unsqueeze(0).expand(candidates.shape[0],-1)
        residual=self.pair(torch.cat([candidates,context],dim=-1)).squeeze(-1)
        delta=self.residual_scale*torch.tanh(residual)
        defer_logit=self.defer(base_context).reshape(())
        return delta,defer_logit

def numpy_predict(model: SupportReranker, x: np.ndarray, context: np.ndarray, raw: np.ndarray, mean: np.ndarray, std: np.ndarray, device='cpu'):
    model.eval(); xn=(np.asarray(x,np.float32)-mean)/std
    with torch.no_grad():
        xx=torch.from_numpy(xn).to(device); cc=torch.from_numpy(np.asarray(context,np.float32)).to(device); delta,dl=model(xx,cc)
    d=delta.detach().cpu().numpy().astype(np.float32); score=np.asarray(raw,np.float32)+d
    return score,float(torch.sigmoid(dl).detach().cpu().item()),d
