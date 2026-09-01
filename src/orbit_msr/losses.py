"""ORBIT-MSR losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.orbit.track_aggregator import geometry_loss


def gate_bce(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target.float())


def gate_margin_loss(logit, target, margin=1.0):
    """Energy-style margin on a binary logit:
    known pushed to +margin, non-known pushed to -margin."""
    y = 2.0 * target.float() - 1.0
    return torch.relu(margin - y * logit).pow(2).mean()


def reuse_bce(logit, target, weight_new=1.0):
    if logit.numel() == 0:
        return torch.zeros((), device=logit.device)
    w = torch.where(target.bool(), torch.tensor(weight_new, device=logit.device),
                    torch.tensor(1.0, device=logit.device))
    return F.binary_cross_entropy_with_logits(logit, target.float(), weight=w)


def known_loss(z, known_protos, target, temperature=0.1):
    logits = torch.mm(z, known_protos.t()) / temperature
    return F.cross_entropy(logits, target)


def novel_metric_loss(z, novel_protos, target, temperature=0.1):
    if novel_protos.shape[0] == 0:
        return torch.zeros((), device=z.device)
    logits = torch.mm(z, novel_protos.t()) / temperature
    return F.cross_entropy(logits, target)
