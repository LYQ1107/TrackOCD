"""Ragged visible-prefix R and a same-parameter genuinely timeless control.

Offsets describe ONLY the supplied observed prefixes, not future track length
or physical/semantic IDs. No observation padding, truncation or support detach.
All graphs remain live; runtime RAM/GPU guards still required.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .raw_preserving_evidence import RawPreservingEvidence, EvidenceTensors, unit
from .differentiable_support_bank import VisiblePrefix


@dataclass(frozen=True)
class FlatPrefixes:
    visual: torch.Tensor
    boxes: torch.Tensor
    quality: torch.Tensor
    elapsed: torch.Tensor
    offsets: torch.Tensor


@dataclass(frozen=True)
class BatchedEvidence:
    raw: torch.Tensor
    embedding: torch.Tensor
    angular_uncertainty: torch.Tensor
    log_prefix_maturity: torch.Tensor
    quality_mean: torch.Tensor
    quality_std: torch.Tensor
    log_elapsed: torch.Tensor
    visible_weights: torch.Tensor
    offsets: torch.Tensor

    def member(self, index):
        first, end = self.offsets[index:index + 2].tolist()
        return EvidenceTensors(self.raw[index], self.embedding[index], self.angular_uncertainty[index],
            self.log_prefix_maturity[index], self.quality_mean[index], self.quality_std[index],
            self.log_elapsed[index], self.visible_weights[first:end])


def pack_visible(prefixes):
    prefixes = tuple(prefixes)
    if not prefixes or len(prefixes) > 4096 or any(type(p) is not VisiblePrefix for p in prefixes):
        raise ValueError('Nonempty bounded numeric visible prefixes only')
    counts = [len(p.quality) for p in prefixes]
    if min(counts) < 1 or sum(counts) > 65536:
        raise ValueError('Whole-call visible bound, no observation truncation')
    values = [torch.cat([getattr(p, field).clone() for p in prefixes])
              for field in ('visual', 'boxes', 'quality', 'elapsed')]
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64, device=values[0].device),
                         torch.tensor(counts, dtype=torch.int64, device=values[0].device).cumsum(0)))
    return FlatPrefixes(*values, offsets)


def batched_evidence(model, flat):
    if not isinstance(model, RawPreservingEvidence) or type(flat) is not FlatPrefixes:
        raise ValueError('Registered R and numeric visible prefix arrays')
    v, b, q, e, o = flat.visual, flat.boxes, flat.quality, flat.elapsed, flat.offsets
    n = len(q); g = len(o) - 1
    if (not 1 <= g <= 4096 or not 1 <= n <= 65536 or v.shape != (n, 768) or b.shape != (n, 4)
            or q.shape != (n,) or e.shape != (n,) or o.shape != (g + 1,)
            or any(x.device != v.device for x in (b, q, e, o))
            or any(x.dtype != torch.float32 for x in (v, b, q)) or e.dtype != torch.int64 or o.dtype != torch.int64
            or o[0] != 0 or o[-1] != n or torch.any(o[1:] <= o[:-1])):
        raise ValueError('Aligned bounded ragged float32/int64 visible layout')
    lengths = o[1:] - o[:-1]
    group = torch.arange(g, device=v.device).repeat_interleave(lengths)
    starts = o[:-1]
    chronology = torch.ones(n, dtype=torch.bool, device=v.device); chronology[starts] = False
    if (torch.any(e[starts] != 0) or torch.any(e[1:][chronology[1:]] <= e[:-1][chronology[1:]])
            or any(not torch.isfinite(x).all() for x in (v, b, q))
            or torch.any((q < 0) | (q > 1)) or torch.any(b[:, 2:] <= b[:, :2])):
        raise ValueError('Finite positive geometry / legal quality / per-prefix elapsed chronology')
    unit(v)
    def sum_groups(value): return torch.segment_reduce(value, 'sum', lengths=lengths)
    def expand(value): return value[group]
    mode = getattr(model, 'representation_mode', 'temporal' if model.temporal else 'pooled_metadata')
    if mode == 'timeless_static':
        # Canonical numeric ordering removes last-bit sum-order tie breakers.
        # Only ordering keys detach; indexed tensors and ALL R gradients stay.
        keys = torch.cat((v, b, q[:, None]), 1).detach().cpu().numpy()
        permutation = []
        for first, end in zip(o[:-1].tolist(), o[1:].tolist()):
            permutation.extend(sorted(range(first, end), key=lambda j: keys[j].tobytes()))
        indices = torch.tensor(permutation, dtype=torch.int64, device=v.device)
        v, b, q = v[indices], b[indices], q[indices]
    base = torch.where(expand(sum_groups(q)) > 1e-12, q, torch.ones_like(q))
    weights = base / expand(sum_groups(base))
    raw = unit(sum_groups(v * weights[:, None]))
    center = (b[:, :2] + b[:, 2:]) / 2; extent = b[:, 2:] - b[:, :2]
    if mode == 'timeless_static':
        spatial = torch.cat((center, extent), 1)
        mean = sum_groups(spatial * weights[:, None])
        variance = sum_groups((spatial - expand(mean)).square() * weights[:, None])
        summary = torch.cat((mean, (variance + 1e-8).sqrt()), 1)  # all eight non-temporal inputs
        gate = 2 * model.gates(summary).squeeze(-1).sigmoid()
        visible_weights = weights
        adapted = v + expand(gate)[:, None] * model.residual(v)
        embedding = unit(sum_groups(adapted * weights[:, None]))
        elapsed_summary = torch.zeros(g, dtype=torch.float32, device=v.device)
    else:
        delta = torch.cat((torch.zeros_like(center[:1]), center[1:] - center[:-1]))
        delta = delta.clone(); delta[starts] = 0
        geometry = torch.cat((center, extent, delta, q[:, None], torch.log1p(e.float())[:, None] / 10), 1)
        if mode == 'temporal':
            scores = model.gates(geometry).squeeze(-1)
            if not torch.isfinite(scores).all(): raise ValueError('Finite temporal gates')
            gated = base * (2 * scores.sigmoid()); totals = sum_groups(gated)
            logg = F.logsigmoid(scores)
            logg = logg - expand(torch.segment_reduce(logg, 'max', lengths=lengths))
            logw = base.log() + logg
            stable = (logw - expand(torch.segment_reduce(logw, 'max', lengths=lengths))).exp()
            fallback = stable / expand(sum_groups(stable))
            visible_weights = torch.where(expand(totals) > 1e-12,
                gated / expand(totals.clamp_min(torch.finfo(torch.float32).tiny)), fallback)
            adapted = v + model.residual(v)
        elif mode == 'pooled_metadata':
            gate = 2 * model.gates(sum_groups(geometry * weights[:, None])).squeeze(-1).sigmoid()
            visible_weights = weights
            adapted = v + expand(gate)[:, None] * model.residual(v)
        else: raise ValueError('Explicit temporal / pooled metadata / timeless variant')
        embedding = unit(sum_groups(adapted * visible_weights[:, None]))
        elapsed_summary = torch.log1p(e[o[1:] - 1].float()) / 10
    uncertainty = (1 - sum_groups((unit(adapted) * expand(embedding)).sum(-1) * visible_weights)).clamp(0, 1)
    maturity = torch.log1p(1 / sum_groups(visible_weights.square())) / 4
    qmean = sum_groups(q) / lengths
    qstd = (sum_groups((q - expand(qmean)).square()) / lengths).sqrt()
    if not torch.isfinite(embedding).all() or not torch.isfinite(visible_weights).all():
        raise ValueError('Finite complete batched evidence')
    if mode == 'timeless_static':
        visible_weights = visible_weights[torch.argsort(indices)]  # caller's original observed row order
    return BatchedEvidence(raw, embedding, uncertainty, maturity, qmean, qstd, elapsed_summary, visible_weights, o)


def reencode_batched_supports(model, candidate_groups, known_flags, binding_counts):
    """Full equal-member prototypes plus current differentiable member evidence.

    The latter enables Raw geometry losses on individual support members, not
    only five class averages. No class/identity arguments to R; candidate-kind
    flags and binding counts remain the same external numeric statistics.
    """
    k = len(candidate_groups)
    if (not 1 <= k <= 4096 or any(not g for g in candidate_groups)
            or known_flags.shape != (k,) or binding_counts.shape != (k,)
            or known_flags.dtype != torch.float32 or binding_counts.dtype != torch.float32
            or known_flags.device != binding_counts.device
            or not torch.isfinite(known_flags).all() or not torch.isfinite(binding_counts).all()
            or torch.any((known_flags != 0) & (known_flags != 1))
            or torch.any(binding_counts < 1) or torch.any(binding_counts != binding_counts.floor())):
        raise ValueError('Complete numeric candidates and kind/binding statistics')
    flat = pack_visible([p for group in candidate_groups for p in group])
    if flat.visual.device != known_flags.device:
        raise ValueError('Same evidence/statistics device')
    members = batched_evidence(model, flat)
    lengths = torch.tensor([len(group) for group in candidate_groups], dtype=torch.int64, device=known_flags.device)
    raw_sum = torch.segment_reduce(members.raw, 'sum', lengths=lengths)
    adapted_sum = torch.segment_reduce(members.embedding, 'sum', lengths=lengths)
    raw, adapted = unit(raw_sum), unit(adapted_sum)
    dispersion = (1 - ((raw_sum / lengths[:, None]) * raw).sum(-1)).clamp(0, 1)
    quality = torch.segment_reduce(members.quality_mean, 'mean', lengths=lengths)
    statistics = torch.stack((known_flags, dispersion, lengths.float(), quality, binding_counts), 1)
    return raw, adapted, statistics, members


class TimelessStaticEvidence(RawPreservingEvidence):
    """Same50113 trainable parameters, no delta/elapsed/sequence-order features.

    Different summary/function class by design, not identical active capacity
    or evidence of matched training. Real seeded controls still required.
    """
    representation_mode = 'timeless_static'

    def __init__(self): super().__init__(temporal=False)

    def forward(self, visual, boxes, quality, elapsed):
        return batched_evidence(self, pack_visible([VisiblePrefix(visual, boxes, quality, elapsed)])).member(0)
