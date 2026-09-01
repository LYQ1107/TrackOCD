"""ORBIT-FC training losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.orbit.track_aggregator import geometry_loss


def bce_gate(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, target.float())


def bce_reuse(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, target.float())


def known_loss(z: torch.Tensor, known_protos: torch.Tensor, target: torch.Tensor,
               temperature: float = 0.1) -> torch.Tensor:
    logits = torch.mm(z, known_protos.t()) / temperature
    return F.cross_entropy(logits, target)


def novel_metric_loss(z: torch.Tensor, novel_protos: torch.Tensor, target: torch.Tensor,
                      temperature: float = 0.1) -> torch.Tensor:
    if novel_protos.shape[0] == 0:
        return torch.zeros((), device=z.device)
    logits = torch.mm(z, novel_protos.t()) / temperature
    return F.cross_entropy(logits, target)


def sem_preservation_loss(z: torch.Tensor, z0: torch.Tensor,
                          frozen_protos: torch.Tensor,
                          temperature: float = 0.1) -> torch.Tensor:
    """KL(p_adapted || p_original) over frozen-DINO known logits.

    Keeps the adapted representation's known semantic distribution aligned
    with the original DINO distribution for train-side known tracks.
    """
    if frozen_protos.shape[0] == 0:
        return torch.zeros((), device=z.device)
    pa = torch.softmax(torch.mm(z, frozen_protos.t()) / temperature, dim=-1)
    po = torch.softmax(torch.mm(z0.detach(), frozen_protos.t()) / temperature, dim=-1)
    return F.kl_div(torch.log(pa.clamp_min(1e-8)), po.clamp_min(1e-8), reduction="batchmean")


def orbit_fc_loss(gate_logit, gate_target, reuse_logit, reuse_target, reuse_mask,
                  z, z0, known_protos, known_target, frozen_protos,
                  novel_protos, novel_target, novel_mask,
                  lambda_reuse=1.0, lambda_known=0.5, lambda_novel=0.5,
                  lambda_geo=0.3, lambda_sem=0.5):
    loss = bce_gate(gate_logit, gate_target)
    if reuse_logit.numel() and reuse_mask.any():
        loss = loss + lambda_reuse * bce_reuse(reuse_logit[reuse_mask],
                                               reuse_target[reuse_mask])
    if known_protos is not None and known_target is not None and known_target.numel():
        loss = loss + lambda_known * known_loss(z, known_protos, known_target)
    if novel_protos is not None and novel_target is not None and novel_target.numel():
        loss = loss + lambda_novel * novel_metric_loss(z, novel_protos, novel_target)
    if frozen_protos is not None and known_target is not None and known_target.numel():
        loss = loss + lambda_sem * sem_preservation_loss(z, z0, frozen_protos)
    loss = loss + lambda_geo * geometry_loss(z, z0)
    return loss
