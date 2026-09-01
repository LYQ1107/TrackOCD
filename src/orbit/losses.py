"""ORBIT training losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.orbit.track_aggregator import geometry_loss


def action_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)


def known_loss(z: torch.Tensor, known_protos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = torch.mm(z, known_protos.t()) / 0.1
    return F.cross_entropy(logits, target)


def novel_metric_loss(z: torch.Tensor, novel_protos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Supervised prototype assignment for pseudo-novel tracks."""
    if novel_protos.shape[0] == 0:
        return torch.zeros((), device=z.device)
    logits = torch.mm(z, novel_protos.t()) / 0.1
    return F.cross_entropy(logits, target)


def orbit_loss(action_logits, action_target, z, z0, known_protos=None, known_target=None,
               novel_protos=None, novel_target=None, lambda_known=0.5, lambda_novel=0.5,
               lambda_geo=0.3):
    loss = action_loss(action_logits, action_target)
    if known_protos is not None and known_target is not None:
        loss = loss + lambda_known * known_loss(z, known_protos, known_target)
    if novel_protos is not None and novel_target is not None:
        loss = loss + lambda_novel * novel_metric_loss(z, novel_protos, novel_target)
    loss = loss + lambda_geo * geometry_loss(z, z0)
    return loss
