from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

class SupportConditionedEncoder(nn.Module):
    """Small causal support-set encoder; metadata never enters this module."""
    def __init__(self, input_dim: int = 768, embedding_dim: int = 256):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, embedding_dim))
        self.support = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, embedding_dim))
        self.fuse = nn.Sequential(nn.LayerNorm(embedding_dim*2), nn.Linear(embedding_dim*2, 256), nn.GELU(), nn.Linear(256, embedding_dim))
    def encode(self, x):
        return F.normalize(self.query(x.float()), dim=-1)
    def forward(self, query, support, support_mask=None):
        q = self.encode(query)
        s = F.normalize(self.support(support.float()), dim=-1)
        if support_mask is None: support_mask = torch.ones(s.shape[:-1], device=s.device, dtype=torch.bool)
        m = support_mask.bool().unsqueeze(-1)
        denom = m.sum(1).clamp_min(1)
        ctx = (s*m.float()).sum(1)/denom
        ctx = F.normalize(ctx, dim=-1)
        z = F.normalize(self.fuse(torch.cat([q,ctx],-1)), dim=-1)
        pair = torch.sum(q[:,None,:]*s, dim=-1).masked_fill(~support_mask.bool(), -1e4)
        return {'embedding': z, 'query_embedding': q, 'support_embeddings': s, 'pair_scores': pair}
