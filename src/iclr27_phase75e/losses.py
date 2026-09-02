"""Fixed Phase75E training objective; no loss or margin sweep is allowed."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .pairwise_adapter import pairwise_torch_score, raw_mean_score


def episode_loss(
    adapter: torch.nn.Module,
    query_by_prefix: dict[int, torch.Tensor],
    positive_by_prefix: list[dict[int, torch.Tensor]],
    negative_by_prefix: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the registered 0.5 rank + 1 raw + 1 safety loss.

    Every prefix is evaluated in the same update.  Assignment indices are
    selected from detached similarities, while selected similarities retain
    gradients.  TRAIN metadata never enters this function.
    """
    rank_terms: list[torch.Tensor] = []
    safe_terms: list[torch.Tensor] = []
    recon_terms: list[torch.Tensor] = []
    for prefix, q_raw in query_by_prefix.items():
        q_adapt = adapter(q_raw)
        neg_raw = negative_by_prefix[prefix]
        neg_adapt = adapter(neg_raw)
        recon_terms.extend([F.mse_loss(q_adapt, q_raw), F.mse_loss(neg_adapt, neg_raw)])
        for pos_by_p in positive_by_prefix:
            pos_raw = pos_by_p[prefix]
            pos_adapt = adapter(pos_raw)
            recon_terms.append(F.mse_loss(pos_adapt, pos_raw))
            s_pos = pairwise_torch_score(q_adapt, pos_adapt)
            s_neg = pairwise_torch_score(q_adapt, neg_adapt)
            rank_terms.append(F.softplus(s_neg - s_pos))
            raw_pos = raw_mean_score(q_raw, pos_raw)
            raw_neg = raw_mean_score(q_raw, neg_raw)
            raw_margin = raw_pos - raw_neg
            adapt_margin = s_pos - s_neg
            # Safety only protects relations that are already correct in raw
            # geometry; raw_margin is detached by contract.
            safe_terms.append(F.relu(raw_margin.detach() - adapt_margin) * (raw_margin.detach() > 0).float())
    rank = torch.stack(rank_terms).mean() if rank_terms else q_raw.new_zeros(())
    recon = torch.stack(recon_terms).mean() if recon_terms else q_raw.new_zeros(())
    safe = torch.stack(safe_terms).mean() if safe_terms else q_raw.new_zeros(())
    total = 0.5 * rank + recon + safe
    return total, {
        "loss": float(total.detach().cpu()),
        "rank": float(rank.detach().cpu()),
        "raw_reconstruction": float(recon.detach().cpu()),
        "safe": float(safe.detach().cpu()),
        "raw_positive_fraction": float(torch.stack([((raw_mean_score(query_by_prefix[p], positive_by_prefix[0][p]) - raw_mean_score(query_by_prefix[p], negative_by_prefix[p])) > 0).float() for p in query_by_prefix]).mean().detach().cpu()),
    }
