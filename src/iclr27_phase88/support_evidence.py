"""Eight-dimensional support evidence computed from real memory candidates."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_support_evidence(query: Tensor, prototypes: Tensor, prototype_mask: Tensor,
                             state_stats: Tensor, quality: Tensor,
                             best_streak: int | Tensor) -> Tensor:
    """Return [B,8] raw/candidate/history features without IDs or labels."""
    # query [B,D], prototypes [B,S,K,D], mask [B,S,K]
    sim = torch.einsum("bd,bskd->bsk", F.normalize(query, dim=-1), F.normalize(prototypes, dim=-1))
    masked = sim.masked_fill(~prototype_mask, -1e4)
    state_best = masked.max(dim=-1).values
    valid_state = prototype_mask.any(dim=-1)
    state_values = state_best.masked_fill(~valid_state, -1e4)
    top = torch.topk(state_values, k=min(2, state_values.shape[1]), dim=-1).values if state_values.shape[1] else query.new_full((query.shape[0], 1), -1e4)
    raw_best = top[:, :1]
    raw_second = top[:, 1:2] if top.shape[1] > 1 else torch.full_like(raw_best, -1e4)
    no_state = ~valid_state.any(dim=-1, keepdim=True)
    raw_best = torch.where(no_state, torch.zeros_like(raw_best), raw_best)
    raw_second = torch.where(no_state, torch.zeros_like(raw_second), raw_second)
    margin = raw_best - raw_second
    count_norm = valid_state.float().sum(dim=-1, keepdim=True) / 16.0
    source_len = state_stats[..., 1].masked_fill(~valid_state, 0.0).sum(dim=-1, keepdim=True) / 16.0
    variance = state_stats[..., 3].masked_fill(~valid_state, 0.0).mean(dim=-1, keepdim=True) if state_stats.shape[1] else query.new_zeros((query.shape[0], 1))
    quality = quality.reshape(-1, 1).float()
    streak = torch.as_tensor(best_streak, device=query.device, dtype=query.dtype).reshape(-1, 1) / 16.0
    return torch.cat([raw_best, raw_second, margin, count_norm, source_len, variance, quality, streak], dim=-1)
