"""EXTERNAL legal policy-pseudo Known-rejection loss prerequisite.

Only the existing Known-derived policy-pseudo family may reject offline fit
Known candidates. This is complementary set supervision, NOT a GT-positive
Existing/NEW/WAIT state target, unknown-as-background, memory repair, Novel
vocabulary, fitted calibration or proof of generalization. No model receives
the family/GT plan; caller must gate source/authority/data before real training.
"""
import torch
from .candidate_training_loss import candidate_supervised_loss


def complementary_known_rejection(logits, bindings, query_family):
    bindings = tuple(bindings)
    if query_family not in (None, 'Known', 'policy_pseudo'):
        raise ValueError('Only authorized fit/policy Known-derived external training families')
    if (logits.dtype != torch.float32 or logits.ndim != 1 or len(logits) != len(bindings) + 2
            or not torch.isfinite(logits).all()):
        raise ValueError('Finite aligned actual candidate plus NEW/WAIT logits')
    named = []
    for group in bindings:
        kinds = {row.kind for row in group}
        if not group or len(kinds) != 1 or not kinds <= {'KNOWN', 'EXISTING'}:
            raise ValueError('Actual family-pure numeric binding rows, no guessed mixed-row targets')
        named.append(kinds == {'KNOWN'})
    if query_family != 'policy_pseudo' or not any(named):
        return None
    # We know only that the query is not one of the frozen fit Known classes.
    # Unknown Existing candidates are NOT declared GT-negative or individually
    # GT-positive. Their relative target/ordering remains unspecified.
    reject = torch.tensor((*named, False, False), dtype=torch.bool, device=logits.device)
    return torch.logsumexp(logits.sort().values, 0) - torch.logsumexp(logits[~reject].sort().values, 0)


def complementary_candidate_loss(logits, plan, bindings, *, coefficient=.25):
    if type(coefficient) is not float or coefficient != .25:
        raise ValueError('Fixed source-prerequisite coefficient, no parameter search')
    base = candidate_supervised_loss(logits, plan)
    reject = complementary_known_rejection(logits, bindings, plan.query_family)
    if reject is None: return base
    return coefficient * reject if base is None else base + coefficient * reject
