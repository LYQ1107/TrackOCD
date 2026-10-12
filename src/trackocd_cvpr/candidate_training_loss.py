"""External loss on actual prediction-owned history; no oracle memory repair.

Only fit/policy existing Known labels may be supplied by the registered caller.
This is a supervised PURE-history objective, not full semantic truth, a deployed
purity feature or a certificate that an observed state generalizes to Novel.
The numeric scorer/controller never receives this history or its labels.
"""
from dataclasses import dataclass
from types import MappingProxyType

import torch


def lawful_label(target):
    kind, binding = target
    if kind not in ('Known', 'policy_pseudo'):
        if binding is not None: raise ValueError('Masked targets must not carry invented semantic bindings')
        return None
    if type(binding) is not int or binding < 0: raise ValueError('External eligible loss binding only')
    return kind, binding


@dataclass(frozen=True, slots=True)
class PredictedCommitLabel:
    # PRIVATE loss/evaluator record, never policy input or a bank write command.
    video: int
    route: str
    frame: int
    token: str
    target: tuple | None


class ExternalPredictionHistory:
    """Append actual anonymous outputs after atomic frame commit, never fix them.

    Snapshot BEFORE the current frame's decisions/loss. Repeated physical
    prefixes do not add independent semantic votes, but every actual historical
    commit remains in the append-only record. A moved member cannot erase old
    token pollution in this strict-history loss. Unknown provenance stays None.
    """
    def __init__(self, *, max_commits=65536):
        if type(max_commits) is not int or max_commits < 1: raise ValueError('Positive history bound')
        self._records = (); self._tokens = {}; self._bound = max_commits

    def snapshot(self):
        return MappingProxyType({token: tuple(sorted(rows.items())) for token, rows in self._tokens.items()})

    def records(self):
        return self._records

    def observe_frame(self, video, result, external_targets):
        records = []; tokens = {token: dict(rows) for token, rows in self._tokens.items()}
        for route, event in result.routed_predictions():
            if event.action not in ('NEW', 'EXISTING'): continue
            if event.token is None: raise ValueError('Actual anonymous action requires its own output token')
            target = lawful_label(external_targets.get((video, route), ('no_retained_semantic_truth', None)))
            key = (video, route); rows = tokens.setdefault(event.token, {})
            if key in rows and rows[key] != target: raise ValueError('External physical label cannot change retroactively')
            rows[key] = target
            records.append(PredictedCommitLabel(video, route, result.frame_index, event.token, target))
        if len(self._records) + len(records) > self._bound: raise ValueError('History bound, no pollution pruning')
        self._tokens = tokens; self._records += tuple(records)


@dataclass(frozen=True, slots=True)
class CandidateLossPlan:
    valid: tuple[bool, ...]
    positive: tuple[bool, ...]
    reason: str
    query_family: str | None
    positive_has_other_physical_member: bool = False
    positive_has_other_video_member: bool = False


def prediction_owned_loss_plan(bindings, query_keys, external_targets, prior_history):
    """Plan after model output, using ONLY its immutable pre-frame history.

    Known queries target exposed uniquely-resolvable offline Known candidates.
    Policy-pseudo queries target verified pure predicted states or NEW when all
    available state histories are semantically verifiable and no pure match
    exists. Unknown candidates are masked, never declared negatives. If there
    is no pure match and an uncertain candidate, the entire pseudo query loss
    is masked: NEW is not justified by a missing/unknown target. Exact binding
    ambiguity yields WAIT, not an oracle numeric-ID choice. Full rollout still
    proceeds for every masked input. Self continuation is distinguished from
    cross-instance/video evidence and is NOT a scientific category-reuse win.
    """
    bindings = tuple(bindings); query_keys = tuple(query_keys); size = len(bindings) + 2
    labels = [lawful_label(external_targets.get(key, ('no_retained_semantic_truth', None))) for key in query_keys]
    if not labels: raise ValueError('One full physical alias group required')
    if any(label is None for label in labels) or len(set(labels)) != 1:
        return CandidateLossPlan((False,) * size, (False,) * size, 'MASKED_QUERY_OR_CONFLICTING_ALIASES', None)
    query = labels[0]; valid = [True] * size; positive = [False] * size
    ambiguous_positive = unknown = other_member = other_video = False
    for i, group in enumerate(bindings):
        if not group: raise ValueError('Nonempty candidate output binding group required')
        match = []; group_unknown = False; provenance = []
        for binding in group:
            if binding.kind == 'KNOWN':
                if type(binding.identifier) is not int: raise ValueError('External named binding')
                match.append(query == ('Known', binding.identifier))
            elif binding.kind == 'EXISTING':
                rows = prior_history.get(binding.identifier, ())
                truth = [target for _, target in rows]
                if not truth or any(target is None for target in truth):
                    group_unknown = True; match.append(False)
                else:
                    match.append(query[0] == 'policy_pseudo' and all(target == query for target in truth))
                    provenance.extend(key for key, _ in rows)
            else:
                raise ValueError('Only actual Known/Existing candidate bindings')
        if group_unknown:
            valid[i] = False; unknown = True  # No unknown-as-background competitor.
        elif len(group) > 1:
            valid[i] = False  # Numeric row cannot resolve the individual output.
            ambiguous_positive |= any(match)
        elif match[0]:
            positive[i] = True
            other_member |= any(key not in query_keys for key in provenance)
            other_video |= any(key[0] not in {v for v, _ in query_keys} for key in provenance)
    if any(positive):
        reason = 'EXPOSED_RESOLVABLE_CORRECT_CANDIDATE'
    elif ambiguous_positive:
        positive[-1] = True; reason = 'WAIT_FOR_UNRESOLVABLE_BINDING_AMBIGUITY'
    elif query[0] == 'policy_pseudo' and not unknown:
        positive[-2] = True; reason = 'NEW_NO_VERIFIED_PURE_CANDIDATE'
    else:
        return CandidateLossPlan((False,) * size, (False,) * size,
            'MASKED_UNCERTAIN_PSEUDO_OR_ABSENT_KNOWN_CANDIDATE', query[0])
    return CandidateLossPlan(tuple(valid), tuple(positive), reason, query[0], other_member, other_video)


def candidate_supervised_loss(logits, plan):
    """Multi-positive log likelihood, exact row-permutation equivariance.

    None means no semantic optimizer contribution, not a physical input drop.
    Canonical reductions keep floating addition order independent of IDs/rows.
    """
    if (logits.shape != (len(plan.valid),) or logits.dtype != torch.float32
            or not torch.isfinite(logits).all() or len(plan.positive) != len(plan.valid)
            or any(p and not v for p, v in zip(plan.positive, plan.valid))):
        raise ValueError('Aligned finite numeric logits and external loss plan')
    if not any(plan.positive): return None
    valid = torch.tensor(plan.valid, dtype=torch.bool, device=logits.device)
    positive = torch.tensor(plan.positive, dtype=torch.bool, device=logits.device)
    return torch.logsumexp(logits[valid].sort().values, 0) - torch.logsumexp(logits[positive].sort().values, 0)
