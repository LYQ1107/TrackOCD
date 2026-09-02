"""Causal-memory-matched TRAIN objective."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_loss(model_out: dict[str, torch.Tensor], positive_indices: list[int], negative_indices: list[int]) -> tuple[torch.Tensor, dict[str, float]]:
    scores = model_out["scores"]
    raw = model_out["raw"]
    if not positive_indices or not negative_indices:
        z = scores.sum() * 0.0
        return z, {"total": 0.0, "listwise": 0.0, "hard_negative": 0.0, "persistence": 0.0, "safety": 0.0, "raw_correct_prefixes": 0.0}
    pos = torch.as_tensor(positive_indices, dtype=torch.long, device=scores.device)
    neg = torch.as_tensor(negative_indices, dtype=torch.long, device=scores.device)
    listwise_terms: list[torch.Tensor] = []
    hard_terms: list[torch.Tensor] = []
    persistence_terms: list[torch.Tensor] = []
    safety_terms: list[torch.Tensor] = []
    raw_correct_count = 0
    for t in range(scores.shape[0]):
        st, rt = scores[t], raw[t]
        listwise_terms.append(-(torch.logsumexp(st[pos], 0) - torch.logsumexp(st, 0)))
        hard_terms.append(F.softplus(torch.max(st[neg]) - torch.max(st[pos])))
        raw_top_pos = bool(int(torch.argmax(rt).item()) in positive_indices)
        raw_margin = torch.max(rt[pos]) - torch.max(rt[neg])
        learned_margin = torch.max(st[pos]) - torch.max(st[neg])
        if raw_top_pos:
            raw_correct_count += 1
            safety_terms.append(F.relu(raw_margin.detach() - learned_margin))
        if t:
            prev_margin = torch.max(scores[t - 1][pos]) - torch.max(scores[t - 1][neg])
            persistence_terms.append(F.relu(prev_margin.detach() - learned_margin))
    listwise = torch.stack(listwise_terms).mean()
    hard = torch.stack(hard_terms).mean()
    persistence = torch.stack(persistence_terms).mean() if persistence_terms else scores.new_zeros(())
    safety = torch.stack(safety_terms).mean() if safety_terms else scores.new_zeros(())
    residual = model_out["residual"].square().mean()
    total = listwise + 0.35 * hard + 0.5 * persistence + 1.5 * safety + 0.02 * residual
    return total, {"total": float(total.detach().cpu()), "listwise": float(listwise.detach().cpu()), "hard_negative": float(hard.detach().cpu()), "persistence": float(persistence.detach().cpu()), "safety": float(safety.detach().cpu()), "residual": float(residual.detach().cpu()), "raw_correct_prefixes": float(raw_correct_count / scores.shape[0])}

