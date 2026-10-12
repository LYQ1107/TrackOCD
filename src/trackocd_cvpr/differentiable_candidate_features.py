"""R/C tensor bridge; retrieval, bindings and loss supervision stay external.

No GT, token or semantic ID parameters. Candidate prototypes must derive from
lawful visible support, never oracle correspondence/purity. This helper does
not integrate the NumPy controller or implement a full training/evaluation run.
"""
import torch
from src.trackocd_cvpr.raw_preserving_evidence import unit

FEATURE_COUNT = 14


def candidate_tensors(query, candidate_raw, candidate_adapted, visible_statistics):
    """Stats: Known flag, dispersion, group count, quality, binding count.

    Fourteen features extend the Raw control's11 with Raw cosine, adapt-minus-
    Raw agreement and adapted margin. Index0 is adapted (not Raw) similarity.
    No labels, individual IDs, evaluator roles or future lengths enter here.
    """
    count = len(candidate_raw)
    if (candidate_raw.shape != (count, 768) or candidate_adapted.shape != (count, 768)
            or visible_statistics.shape != (count, 5)):
        raise ValueError('Aligned candidate vectors and numeric support statistics required')
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all()
           for value in (candidate_raw, candidate_adapted, visible_statistics)):
        raise ValueError('Finite float32 numeric candidates required')
    if count:
        flags, dispersion, support, quality, aliases = visible_statistics.unbind(1)
        if (torch.any((flags != 0) & (flags != 1)) or torch.any((dispersion < 0) | (dispersion > 1))
                or torch.any(support < 0) or torch.any(support != support.floor())
                or torch.any((quality < 0) | (quality > 1)) or torch.any(aliases < 1)
                or torch.any(aliases != aliases.floor())):
            raise ValueError('Invalid visible support statistics; not GT purity')
        raw_cos = (unit(candidate_raw) @ query.raw).clamp(-1, 1)
        adapted_cos = (unit(candidate_adapted) @ query.embedding).clamp(-1, 1)
        best = adapted_cos.max()
        shared = torch.stack((query.angular_uncertainty, query.log_prefix_maturity,
                              query.quality_mean, query.quality_std, query.log_elapsed))
        base = torch.stack((adapted_cos, flags, dispersion, torch.log1p(support) / 8, quality), 1)
        rows = torch.cat((base, shared.expand(count, -1), torch.log1p(aliases)[:, None] / 8,
                          raw_cos[:, None], (adapted_cos - raw_cos)[:, None], (best - adapted_cos)[:, None]), 1)
    else:
        rows = query.embedding.new_empty((0, FEATURE_COUNT))
        best = query.embedding.new_tensor(-1.)
    context = torch.stack((query.angular_uncertainty, query.log_prefix_maturity,
                           query.quality_mean, query.quality_std, query.log_elapsed, best))
    if not torch.isfinite(rows).all() or not torch.isfinite(context).all():
        raise ValueError('Nonfinite differentiable candidate/query evidence')
    return rows, context
