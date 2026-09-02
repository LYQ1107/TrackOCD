"""Phase76AR fixed task/safety/gate objective."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def teacher_use(raw_scores: torch.Tensor, positive_indices: list[int], relation_features: list[dict[str, torch.Tensor]], negative_indices: list[int]) -> float:
    """TRAIN-only no-harm teacher label from frozen pair evidence.

    A raw-correct query abstains.  For a raw-wrong query the relation teacher
    may open the gate only when the best matched-cosine evidence ranks a
    positive over every negative.  Category labels are used outside this
    function to construct ``positive_indices`` and never enter model inputs.
    """
    if not positive_indices or not negative_indices:
        return 0.0
    if int(torch.argmax(raw_scores).item()) in set(positive_indices):
        return 0.0
    evidence = torch.stack([x["quality_features"][:, 0].mean() if x["quality_features"].numel() else raw_scores.new_tensor(-1.0) for x in relation_features])
    pos = evidence[positive_indices].max(); neg = evidence[negative_indices].max()
    return float((pos > neg).item())


def ar_loss(outputs: dict[str, torch.Tensor], positive_indices: list[int], negative_indices: list[int], teacher: float, task_scale: float = 1.0, safe_scale: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    scores = outputs["final"]; raw = outputs["raw"]
    if not positive_indices or not negative_indices:
        z = scores.sum() * 0.0
        return z, {"task": 0.0, "safe": 0.0, "gate": 0.0, "residual": 0.0, "total": 0.0, "teacher": teacher}
    raw_correct = int(torch.argmax(raw).item()) in set(positive_indices)
    task = -(torch.logsumexp(scores[positive_indices], dim=0) - torch.logsumexp(scores, dim=0))
    task_weight = 0.25 if raw_correct else 1.0
    final_pos = scores[positive_indices].max()
    safe_terms: list[torch.Tensor] = []
    if raw_correct:
        raw_pos = raw[positive_indices].max().detach()
        for neg_idx in negative_indices:
            raw_margin = raw_pos - raw[neg_idx].detach()
            final_margin = final_pos - scores[neg_idx]
            safe_terms.append(F.relu(raw_margin - final_margin))
    safe = torch.stack(safe_terms).mean() if safe_terms else scores.new_zeros(())
    # ``score_bank`` repeats the query-level gate once per candidate so the
    # caller can align tensors.  BCE is a query-level target; reduce the
    # repeated values to one scalar rather than broadcasting a scalar label.
    gate_logit = outputs["bank_gate_logit"].reshape(-1).mean()
    target = gate_logit.new_tensor(float(teacher))
    gate = F.binary_cross_entropy_with_logits(gate_logit, target)
    residual = (outputs["delta_bounded"] ** 2).mean()
    total = task_weight * task / max(task_scale, 1e-6) + safe / max(safe_scale, 1e-6) + 0.5 * gate + 0.01 * residual
    return total, {"task": float(task.detach().cpu()), "safe": float(safe.detach().cpu()), "gate": float(gate.detach().cpu()), "residual": float(residual.detach().cpu()), "total": float(total.detach().cpu()), "teacher": teacher, "raw_correct": float(raw_correct)}
