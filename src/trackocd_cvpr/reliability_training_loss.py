"""External legal support-agreement targets, NEVER deployed purity features.

Caller must gate fit/policy's existing Known labels before entry. Unavailable
or conflicting alias truth masks the loss, not background/Novel/negatives.
Current active support agreement is NOT strict historical lineage purity.
The controller never receives these plans or ground-truth-driven actions.
"""
from dataclasses import dataclass
import hashlib
import torch
from .candidate_training_loss import lawful_label


def prediction_owned_support_targets(bank, token, external_targets, epoch_to_video):
    """Join latest REAL commits to numeric Raw groups outside forward.

    Identical payloads in different scopes can be indistinguishable here. If
    any scope has unknown/conflicting truth, ALL copies are conservatively
    unscorable; never pick a numeric route/video ID to invent a target.
    Historical mistakes remain in bank.commit_history, not GT repaired.
    """
    rows = [row for row in bank.raw_support_snapshot() if row.token == token]
    if len(rows) != 1 or not rows[0].payloads:
        raise ValueError('Actual nonempty prediction-owned support required')
    latest = {}
    for commit in bank.commit_history(): latest[(commit.video_epoch, commit.routing_id)] = commit
    by_payload = {}
    for (epoch, route), commit in latest.items():
        if commit.token != token: continue
        if epoch not in epoch_to_video: raise ValueError('Complete actual played-video provenance required')
        video = epoch_to_video[epoch]
        target = lawful_label(external_targets.get((video, route), ('no_retained_semantic_truth', None)))
        by_payload.setdefault(commit.visible_input_sha256, []).append(target)
    result = []
    for payload in rows[0].payloads:
        labels = by_payload.get(hashlib.sha256(payload).hexdigest(), ())
        if not labels: raise ValueError('Raw snapshot lacks actual latest-commit provenance')
        result.append(labels[0] if all(label is not None and label == labels[0] for label in labels) else None)
    return tuple(result)


@dataclass(frozen=True)
class SupportAgreementPlan:
    targets: tuple[float | None, ...]
    external_class_bindings: tuple[tuple | None, ...]
    reason: str


def support_agreement_plan(external_group_targets):
    """Soft target = fraction of OTHER groups sharing the member's label.

    No oracle majority selection, label-driven pruning, or positive singleton.
    Every peer must be semantically available to measure the whole-support
    denominator. Labels affect loss ONLY, not numeric coherence or lifecycle.
    """
    labels = tuple(external_group_targets)
    if not 1 <= len(labels) <= 512:
        raise ValueError('Bounded whole support; no favorable subset')
    for label in labels:
        if label is not None and lawful_label(label) != label:
            raise ValueError('Eligible fit/policy external label or explicit unknown')
    if len(labels) == 1:
        return SupportAgreementPlan((None,), labels, 'MASKED_SINGLETON_NO_OTHER_SUPPORT')
    if any(label is None for label in labels):
        return SupportAgreementPlan((None,) * len(labels), labels, 'MASKED_UNKNOWN_OR_CONFLICTING_SUPPORT')
    targets = tuple(sum(other == label for j, other in enumerate(labels) if j != i) / (len(labels) - 1)
                    for i, label in enumerate(labels))
    return SupportAgreementPlan(targets, labels, 'EXTERNAL_FULL_OTHER_GROUP_AGREEMENT')


def support_agreement_losses(logits, plan):
    """Class-labelled terms for external macro reduction; no optimizer here."""
    if (logits.dtype != torch.float32 or logits.shape != (len(plan.targets),)
            or not torch.isfinite(logits).all() or len(plan.external_class_bindings) != len(plan.targets)):
        raise ValueError('Aligned finite numeric logits and external targets')
    result = []
    for logit, target, label in zip(logits, plan.targets, plan.external_class_bindings):
        if target is None: continue
        if label is None or not 0 <= target <= 1:
            raise ValueError('Legal external supervised agreement target')
        result.append((label, torch.nn.functional.binary_cross_entropy_with_logits(logit, logit.new_tensor(target))))
    return tuple(result)
