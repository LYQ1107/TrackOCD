import torch
from torch import nn

class MonotonicRawReranker(nn.Module):
    """Raw cosine plus bounded residual from causal pair metadata."""
    def __init__(self, meta_dim=10):
        super().__init__(); self.residual=nn.Sequential(nn.LayerNorm(meta_dim),nn.Linear(meta_dim,32),nn.GELU(),nn.Linear(32,1)); self.scale=nn.Parameter(torch.tensor(-2.0))
    def forward(self, raw, meta):
        bound=0.20*torch.sigmoid(self.scale); return raw + bound*torch.tanh(self.residual(meta)).squeeze(-1)
