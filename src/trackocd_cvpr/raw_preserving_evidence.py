"""Identity-start observed-prefix evidence, no class IDs or supervision inputs.

Source/synthetic prerequisite, not a fitted Novel-generalizable representation.
Inputs must already contain only current/past observations. Geometry gates are
not calibrated reliability. Pooled-gate and temporal modes share parameters,
not function class or an empirical matched-training result. The pooled control
still has pooled motion/time metadata: it is not a pure timeless static model.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EvidenceTensors:
    raw: torch.Tensor
    embedding: torch.Tensor
    angular_uncertainty: torch.Tensor
    log_prefix_maturity: torch.Tensor
    quality_mean: torch.Tensor
    quality_std: torch.Tensor
    log_elapsed: torch.Tensor
    visible_weights: torch.Tensor


def unit(value):
    norm = value.norm(dim=-1, keepdim=True)
    if not torch.isfinite(value).all() or not torch.isfinite(norm).all() or torch.any(norm <= 1e-12):
        raise ValueError('Finite nonzero visible evidence required')
    return value / norm


class RawPreservingEvidence(nn.Module):
    def __init__(self, *, temporal=True):
        super().__init__()
        if type(temporal) is not bool:
            raise ValueError('Explicit static/temporal source configuration required')
        self.temporal = temporal
        self.residual = nn.Sequential(nn.Linear(768, 32), nn.GELU(), nn.Linear(32, 768))
        self.gates = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.gates[-1].weight)
        nn.init.zeros_(self.gates[-1].bias)

    def forward(self, visual, boxes, quality, elapsed):
        count = len(quality)
        if (count < 1 or visual.shape != (count, 768) or boxes.shape != (count, 4)
                or quality.shape != (count,) or elapsed.shape != (count,)):
            raise ValueError('Aligned observed-prefix tensor shapes required')
        if (any(value.dtype != torch.float32 for value in (visual, boxes, quality))
                or elapsed.dtype != torch.int64 or elapsed[0] != 0
                or torch.any(elapsed[1:] <= elapsed[:-1])):
            raise ValueError('Float32 values and exact elapsed chronology required')
        if (any(not torch.isfinite(value).all() for value in (visual, boxes, quality))
                or torch.any((quality < 0) | (quality > 1))
                or torch.any(boxes[:, 2:] <= boxes[:, :2])):
            raise ValueError('Invalid visible geometry/quality/evidence')
        unit(visual)  # No zero descriptor hidden by a positive-quality neighbor.
        base_weights = quality if quality.sum() > 1e-12 else torch.ones_like(quality)
        weights = base_weights / base_weights.sum()
        raw = unit((visual * weights[:, None]).sum(0))
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        delta = torch.cat((torch.zeros_like(center[:1]), center[1:] - center[:-1]))
        geometry = torch.cat((center, boxes[:, 2:] - boxes[:, :2], delta,
                              quality[:, None], torch.log1p(elapsed.float())[:, None] / 10), 1)
        residual = self.residual(visual)
        if self.temporal:
            scores = self.gates(geometry).squeeze(-1)
            if not torch.isfinite(scores).all():
                raise ValueError('Nonfinite learned geometry gate')
            gates = 2 * scores.sigmoid()
            visible_weights = base_weights * gates
            if visible_weights.sum() <= 1e-12:
                # Same positive-weight formula in log space, not silent member
                # filtering or a GT-driven reliability repair.
                log_gates = F.logsigmoid(scores)
                log_gates = log_gates - log_gates.max()
                visible_weights = (base_weights.log() + log_gates).softmax(0)
            else:
                visible_weights = visible_weights / visible_weights.sum()
            adapted = visual + residual
            embedding = unit((adapted * visible_weights[:, None]).sum(0))
        else:
            gate = 2 * self.gates((geometry * weights[:, None]).sum(0)).squeeze().sigmoid()
            visible_weights = weights
            adapted = visual + gate * residual
            embedding = unit((adapted * weights[:, None]).sum(0))
        uncertainty = (1 - (unit(adapted) @ embedding * visible_weights).sum()).clamp(0, 1)
        maturity = torch.log1p(1 / visible_weights.square().sum()) / 4
        return EvidenceTensors(raw, embedding, uncertainty, maturity, quality.mean(),
            quality.std(unbiased=False), torch.log1p(elapsed[-1].float()) / 10, visible_weights)


def raw_geometry_loss(raw, adapted):
    """Raw teacher Gram penalty; no category labels or efficacy claim."""
    if raw.ndim != 2 or not len(raw) or adapted.shape != raw.shape or raw.shape[1] != 768:
        raise ValueError('Aligned Raw/adapted 768D batches required')
    teacher = unit(raw).detach()
    student = unit(adapted)
    return (student @ student.T - teacher @ teacher.T).square().mean()
