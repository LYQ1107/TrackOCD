"""Reencode visible Raw support under current R weights without NumPy detach.

Caller supplies already lawful, latest-per-member, alias-deduplicated visible
groups. This numeric bridge cannot attest their provenance or do retrieval,
transactional writes, lineage evaluation, GT purity repair or model training.
"""
from dataclasses import dataclass
import torch
from .raw_preserving_evidence import RawPreservingEvidence, unit


@dataclass(frozen=True)
class VisiblePrefix:
    visual: torch.Tensor
    boxes: torch.Tensor
    quality: torch.Tensor
    elapsed: torch.Tensor


def reencode_supports(model, candidate_groups, known_flags, binding_counts):
    """Equal-group prototype, raw dispersion/quality and differentiable R.

No IDs, tokens, class/evaluator labels, GT purity or whole trajectory length.
Known flag refers to candidate kind, NOT query truth. Binding counts refer to
feature-indistinguishable output bindings, NOT independent support certainty.
"""
    if not isinstance(model, RawPreservingEvidence) or len(candidate_groups) > 4096:
        raise ValueError('Bounded registered evidence model and numeric candidates required')
    n = len(candidate_groups)
    if (known_flags.shape != (n,) or binding_counts.shape != (n,)
            or known_flags.dtype != torch.float32 or binding_counts.dtype != torch.float32
            or not torch.isfinite(known_flags).all() or not torch.isfinite(binding_counts).all()
            or torch.any((known_flags != 0) & (known_flags != 1))
            or torch.any(binding_counts < 1) or torch.any(binding_counts != binding_counts.floor())
            or known_flags.device != binding_counts.device):
        raise ValueError('Finite candidate-kind/binding-count statistics only')
    raw_rows, adapted_rows, stats = [], [], []
    total_groups, total_observations = 0, 0
    for index, groups in enumerate(candidate_groups):
        if not groups:
            raise ValueError('Nonempty already-deduplicated visible support required')
        total_groups += len(groups)
        if total_groups > 4096:
            raise ValueError('Per-query support reencoding bound; no silent member pruning')
        evidence = []
        for prefix in groups:
            if type(prefix) is not VisiblePrefix or prefix.visual.device != known_flags.device:
                raise ValueError('Numeric visible prefix only, same device as statistics')
            total_observations += len(prefix.quality)
            if total_observations > 16384:
                raise ValueError('Per-query visible reencoding bound, not observation truncation or full benchmark scalability')
            # Copy inputs so an external mutation of the original snapshot
            # cannot invalidate saved tensors for this loss's backward pass.
            evidence.append(model(prefix.visual.clone(), prefix.boxes.clone(),
                                  prefix.quality.clone(), prefix.elapsed.clone()))
        raws = torch.stack([e.raw for e in evidence])
        raw = unit(raws.mean(0))
        adapted = unit(torch.stack([e.embedding for e in evidence]).mean(0))
        dispersion = (1 - (raws @ raw).mean()).clamp(0, 1)
        quality = torch.stack([e.quality_mean for e in evidence]).mean()
        raw_rows.append(raw); adapted_rows.append(adapted)
        stats.append(torch.stack((known_flags[index], dispersion, raw.new_tensor(len(groups)),
                                  quality, binding_counts[index])))
    if not n:
        return (known_flags.new_empty((0, 768)), known_flags.new_empty((0, 768)), known_flags.new_empty((0, 5)))
    return torch.stack(raw_rows), torch.stack(adapted_rows), torch.stack(stats)
