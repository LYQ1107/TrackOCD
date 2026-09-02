"""Registered Phase76A task/safety/residual losses."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def _margin(scores: torch.Tensor, raw_scores: torch.Tensor, pos_idx: list[int], neg_idx: list[int]) -> tuple[torch.Tensor, bool]:
    if not pos_idx or not neg_idx:
        return scores.new_zeros(()), False
    raw_top = int(torch.argmax(raw_scores).item())
    raw_correct = raw_top in set(pos_idx)
    if not raw_correct:
        return scores.new_zeros(()), False
    pos = scores[pos_idx].max()
    # The registered safety margin uses the eight strongest raw negatives.
    neg_order = sorted(neg_idx, key=lambda i: float(raw_scores[i].detach().cpu()), reverse=True)[:8]
    return pos - scores[neg_order].max(), True


def bank_loss(
    outputs_by_prefix: Iterable[dict[str, torch.Tensor]],
    raw_scores: torch.Tensor,
    pos_idx: list[int],
    neg_idx: list[int],
    task_scale: float,
    safe_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    tasks: list[torch.Tensor] = []; safes: list[torch.Tensor] = []; residuals: list[torch.Tensor] = []
    valid = 0
    for output in outputs_by_prefix:
        scores = output["final"]
        if not pos_idx:
            continue
        tasks.append(-(torch.logsumexp(scores[pos_idx], dim=0) - torch.logsumexp(scores, dim=0)))
        margin, ok = _margin(scores, raw_scores.to(scores), pos_idx, neg_idx)
        if ok:
            raw_margin, _ = _margin(raw_scores.to(scores), raw_scores.to(scores), pos_idx, neg_idx)
            safes.append(F.relu(raw_margin.detach() - margin)); valid += 1
        residuals.append(torch.mean((output["confidence"] * output["delta"]) ** 2))
    if not tasks:
        z = raw_scores.new_zeros((), requires_grad=True); return z, {"task": 0.0, "safe": 0.0, "residual": 0.0}
    task = torch.stack(tasks).mean(); safe = torch.stack(safes).mean() if safes else task.new_zeros(())
    residual = torch.stack(residuals).mean() if residuals else task.new_zeros(())
    total = 0.5 * (task / max(float(task_scale), 1e-6) + safe / max(float(safe_scale), 1e-6)) + 0.01 * residual
    return total, {"task": float(task.detach().cpu()), "safe": float(safe.detach().cpu()), "residual": float(residual.detach().cpu()), "valid_safe_prefixes": float(valid), "total": float(total.detach().cpu())}

