"""Candidate-conditioned numeric interface, with external output bindings.

Raw engineering implementation; no GT, fitted weights or metric selection.
K counts feature-distinguishable groups per family; score ties at the K boundary
expand the pool within a hard bound rather than choosing numeric IDs.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

FEATURE_COUNT = 11
CONTEXT_COUNT = 6


@dataclass(frozen=True, slots=True)
class CandidateBinding:
    kind: str
    identifier: int | str


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    feature_bytes: bytes
    context_bytes: bytes
    bindings: tuple[tuple[CandidateBinding, ...], ...]
    requested_K_per_family: int

    def numeric_inputs(self):
        return (np.frombuffer(self.feature_bytes, dtype='<f4').reshape(-1, FEATURE_COUNT),
                np.frombuffer(self.context_bytes, dtype='<f4'))


def unit(vector):
    value = np.asarray(vector, dtype=np.float32)
    if value.shape != (768,) or not np.isfinite(value).all() or np.linalg.norm(value) <= 1e-12:
        raise ValueError('Finite nonzero common768D evidence required')
    return value / np.linalg.norm(value)


def build_candidates(view, known, state_evidence, *, K=8, max_groups=4096):
    if type(K) is not int or K not in (1, 3, 5, 8) or type(max_groups) is not int or max_groups < 1:
        raise ValueError('Finite registered K and candidate-group bound required')
    z = unit(view.weighted_mean())
    norms = np.linalg.norm(view.visual, axis=1)
    if np.any(norms <= 1e-12) or np.any(view.quality > 1):
        raise ValueError('Angular evidence/quality invalid')
    weights = view.quality.astype(np.float64)
    if weights.sum() <= 1e-12:
        weights = np.ones(len(weights))
    weights /= weights.sum()
    uncertainty = float(np.clip(1 - weights @ ((view.visual / norms[:, None]) @ z), 0, 1))
    maturity = float(np.log1p(1 / np.square(weights).sum()) / 4)
    qmean, qstd = float(view.quality.mean()), float(view.quality.std())
    elapsed = float(np.log1p(view.elapsed_frames[-1]) / 10)
    query = [uncertainty, maturity, qmean, qstd, elapsed]
    grouped = {}
    def add(binding, prototype, dispersion, count, quality):
        sim = float(np.clip(z @ unit(prototype), -1, 1))
        base = np.asarray([sim, float(binding.kind == 'KNOWN'), dispersion,
                           np.log1p(count) / 8, quality, *query], dtype='<f4').tobytes()
        grouped.setdefault(base, []).append(binding)
    for key, vector in known.items():
        if type(key) is not int or key < 0:
            raise ValueError('Offline Known bindings required')
        # No fabricated Known support/quality statistics are available.
        add(CandidateBinding('KNOWN', key), vector, 0., 0, 0.)
    for token, state in state_evidence.items():
        prototype = state.prototype()
        if prototype is not None:
            add(CandidateBinding('EXISTING', token), prototype, state.dispersion,
                state.support_groups, state.mean_quality)
    selected = []
    for known_flag in (1., 0.):
        rows = [(np.frombuffer(base, dtype='<f4'), base, bindings) for base, bindings in grouped.items()
                if np.frombuffer(base, dtype='<f4')[1] == known_flag]
        rows.sort(key=lambda row: (-float(row[0][0]), row[1]))
        if rows:
            cutoff = float(rows[min(K, len(rows)) - 1][0][0])
            selected.extend(row for row in rows if float(row[0][0]) >= cutoff)
    if len(selected) > max_groups:
        raise ValueError('Tie-expanded group bound; no ID-based trimming')
    features, bindings = [], []
    for base_row, _, group in selected:
        features.append(np.append(base_row, np.log1p(len(group)) / 8).astype('<f4'))
        # Only external correspondence order; not exposed to the scorer.
        bindings.append(tuple(sorted(group, key=lambda item: str(item.identifier))))
    values = np.asarray(features, dtype='<f4').reshape(-1, FEATURE_COUNT)
    best = float(values[:, 0].max()) if len(values) else -1.
    context = np.asarray([*query, best], dtype='<f4')
    if not np.isfinite(values).all() or not np.isfinite(context).all():
        raise ValueError('Finite visible candidate features required')
    return CandidateSnapshot(values.tobytes(), context.tobytes(), tuple(bindings), K)


def resolve_scores(snapshot, scores):
    values = np.asarray(scores, dtype=np.float32)
    count = len(snapshot.bindings)
    if values.shape != (count + 2,) or not np.isfinite(values).all():
        raise ValueError('One score per group plus NEW/WAIT required')
    maxima = np.flatnonzero(values == values.max())
    if len(maxima) != 1:
        return 'WAIT', None, 'AMBIGUOUS_MAXIMA'
    index = int(maxima[0])
    if index == count:
        return 'NEW', None, None
    if index == count + 1:
        return 'WAIT', None, None
    choices = snapshot.bindings[index]
    if len(choices) != 1:
        return 'WAIT', None, 'INDISTINGUISHABLE_BINDINGS'
    return choices[0].kind, choices[0].identifier, None


class CandidateScoreMLP(nn.Module):
    """708 parameters, shared row scoring; not checkpoint-compatible with D1/D2.

    New and WAIT depend on numeric query/pool context. Candidate permutation
    permutes only candidate scores; output bindings never enter forward.
    Initialization/gradient tests do not constitute any fit or calibration.
    """
    def __init__(self):
        super().__init__()
        self.rows = nn.Sequential(nn.Linear(FEATURE_COUNT, 32), nn.GELU(), nn.Linear(32, 1))
        self.actions = nn.Sequential(nn.Linear(CONTEXT_COUNT, 32), nn.GELU(), nn.Linear(32, 2))
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, features, context):
        if features.ndim != 2 or features.shape[1] != FEATURE_COUNT or context.shape != (CONTEXT_COUNT,):
            raise ValueError('Numeric candidate/query shapes differ')
        if not torch.isfinite(features).all() or not torch.isfinite(context).all():
            raise ValueError('Nonfinite numeric scorer input')
        if features.dtype != torch.float32 or context.dtype != torch.float32:
            raise ValueError('Registered float32 numeric scoring contract required')
        # Identical matrix layout under row permutation avoids backend last-bit
        # differences becoming an artificial candidate-index tie breaker.
        raw = features.detach().cpu().numpy()
        order = sorted(range(len(raw)), key=lambda index: raw[index].tobytes())
        indices = torch.tensor(order, dtype=torch.long, device=features.device)
        inverse = torch.argsort(indices)
        scale = self.log_temperature.clamp(-4, 4).exp()
        canonical = self.rows(features[indices]).squeeze(-1) / scale
        return torch.cat((canonical[inverse], self.actions(context) / scale))
