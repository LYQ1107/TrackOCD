"""Numeric, past-support-only M prerequisite; NOT trained/calibrated memory.

Each member is compared with OTHER visible support groups, not a prototype
containing itself. Exact aliases/latest prefixes must already be resolved by
the caller's lawful pre-frame bank. Groups are not independent-sample claims.
No query truth, identity, token, role, total/future length or GT purity enters
forward. Lifecycle, revision, losses and causal integration remain separate.
"""
from dataclasses import dataclass
import torch
from torch import nn


# The standalone live-memory bridge has a narrower per-token contract, but
# joint R/C/M candidate formation must also accept the complete offline Known
# support group.  These bounds match the registered tensor bridge and refuse
# the whole group rather than silently pruning it.
MAX_SUPPORT_MEMBERS = 4096
MAX_SUPPORT_OBSERVATIONS = 65536


def member_features(raw, quality, observed_counts):
    """Eight numeric features, with explicit absent/cancelled peer evidence.

Bounds refuse the WHOLE call rather than prune inconvenient members. Raw
vectors are immutable snapshots cloned here; no adapted embedding is cached.
"""
    if (raw.ndim != 2 or raw.shape[1] != 768 or not 1 <= len(raw) <= MAX_SUPPORT_MEMBERS
            or quality.shape != (len(raw),) or observed_counts.shape != (len(raw),)
            or any(a.dtype != torch.float32 for a in (raw, quality, observed_counts))
            or any(a.device != raw.device for a in (quality, observed_counts))
            or any(not torch.isfinite(a).all() for a in (raw, quality, observed_counts))
            or torch.any((quality < 0) | (quality > 1))
            or torch.any(observed_counts < 1)
            or torch.any(observed_counts != observed_counts.floor())
            or observed_counts.sum() > MAX_SUPPORT_OBSERVATIONS):
        raise ValueError('Bounded numeric visible support groups required; no pruning')
    raw, quality, observed_counts = raw.clone(), quality.clone(), observed_counts.clone()
    norms = raw.norm(dim=1)
    if torch.any(norms <= 1e-12):
        raise ValueError('Nonzero Raw group aggregates required')
    raw = raw / norms[:, None]
    n = len(raw)
    other_count = n - 1
    if not other_count:
        zero, one = quality.new_zeros(1), quality.new_ones(1)
        return torch.stack((zero, zero, zero, quality, observed_counts.log1p(),
                            zero, one, zero), dim=1), raw
    # Remove self before BOTH peer direction and disagreement statistics.
    # Float64 sums avoid an N x N x 768 allocation and reduce cancellation.
    sums = (raw.double().sum(dim=0)[None] - raw.double()).to(raw.dtype)
    magnitude = sums.norm(dim=1)
    direction = sums / magnitude.clamp_min(1e-12)[:, None]
    cosine = (raw * direction).sum(dim=1).clamp(-1, 1)
    pair_mean = ((raw * sums).sum(dim=1) / other_count).clamp(-1, 1)
    dispersion = (1 - magnitude / other_count).clamp(0, 1)
    features = torch.stack((cosine, pair_mean, dispersion, quality,
        observed_counts.log1p(), quality.new_full((n,), float(other_count)).log1p(),
        quality.new_zeros(n), (magnitude <= 1e-12).to(torch.float32)), dim=1)
    return features, raw


class MemberReliabilityMLP(nn.Module):
    """161 parameters; output logits are NOT calibrated probabilities.

Pooled control has identical nominal parameters but intentionally no relative
member discrimination. All weights start equal; hidden layers can learn on
the first update through the nonzero final-layer gradient.
"""
    def __init__(self, *, pooled_control=False):
        super().__init__()
        if type(pooled_control) is not bool:
            raise ValueError('Explicit relative/pooled numeric control')
        self.pooled_control = pooled_control
        self.rows = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.rows[-1].weight)
        nn.init.zeros_(self.rows[-1].bias)

    def forward(self, features):
        if (features.ndim != 2 or features.shape[1] != 8 or not 1 <= len(features) <= MAX_SUPPORT_MEMBERS
                or features.dtype != torch.float32 or not torch.isfinite(features).all()):
            raise ValueError('Finite bounded numeric member features only')
        value = features.clone()
        if self.pooled_control:
            value = value.mean(dim=0, keepdim=True).expand_as(value)
        return self.rows(value).squeeze(-1)


@dataclass(frozen=True)
class ReliabilityEvidence:
    features: torch.Tensor
    logits: torch.Tensor
    weights: torch.Tensor
    prototype: torch.Tensor | None


def weighted_support(model, raw, quality, observed_counts):
    """Positive-weight whole-support control, not quarantine or purity repair.

The fixed 1e-3 floor prevents numeric zero weights/implicit eviction. No
threshold is selected, members removed, or GT-consistency asserted. A zero
resultant returns None, never an arbitrary direction or fabricated prototype.
"""
    if type(model) is not MemberReliabilityMLP:
        raise ValueError('Registered numeric member reliability module required')
    features, vectors = member_features(raw, quality, observed_counts)
    logits = model(features)
    weights = 1e-3 + (1 - 1e-3) * logits.sigmoid()
    total = (vectors * weights[:, None]).sum(dim=0)
    magnitude = total.norm()
    prototype = None if magnitude <= 1e-12 else total / magnitude
    return ReliabilityEvidence(features, logits, weights, prototype)
