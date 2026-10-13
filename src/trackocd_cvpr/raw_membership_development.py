"""Frozen scalar controls on the unchanged full causal controller.

Private external truth is used only AFTER input-only scores. No fitting,
semantic names, original annotations, sample selection or deployment claim.
"""
from collections import Counter, defaultdict
import numpy as np
import torch
from torch import nn

from .C_development_rollout import readonly_C_stream
from .causal_observation_router import CurrentObservationRouter
from .policy_training_data import policy_current_frames
from .raw_known_membership_control import (
    calibrated_raw_scores, max_known_cosine, membership_diagnostics)


class FrozenMembershipScorer(nn.Module):
    """Adapter ONLY; Top8_conditioned is the unchanged numeric input interface.

    It does not make this two-parameter head a learned 708-parameter C model.
    No labels, role IDs or controller/bank access in the score callback.
    """
    def __init__(self, head):
        super().__init__()
        if head.training or any(p.requires_grad for p in head.parameters()):
            raise ValueError('Frozen own formal membership head required')
        self.head = head
        self.eval().requires_grad_(False)

    def forward(self, features, context):
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError('Read-only development, no parameter update')
        if features.device.type != 'cpu' or context.device.type != 'cpu':
            raise ValueError('One CPU scorer, no CUDA')
        scores = calibrated_raw_scores(features.detach().numpy(), context.detach().numpy(), model=self.head)
        return torch.from_numpy(scores)


def readonly_membership_stream(videos, labels, known, bindings, cfg, order, guard,
                               *, head, rename_and_reverse=False):
    # variant_numeric_features(Top8_conditioned) is an exact float32 copy;
    # common resolver/router/bank/GT audit/timing and probes stay unchanged.
    if cfg['K_per_family']['Top8_conditioned'] != 8:
        raise ValueError('Same fixed Raw Top8 candidate interface')
    return readonly_C_stream(videos, labels, known, bindings, cfg, 'Top8_conditioned',
        order, guard, model=FrozenMembershipScorer(head), rename_and_reverse=rename_and_reverse)


def external_group_target(labels, video, aliases):
    values = [labels[(video, route)] for route in aliases]
    if not values or len(set(values)) != 1:
        return None
    kind, category = values[0]
    if kind not in ('Known', 'declared_pseudoNovel'):
        if category is not None:
            raise ValueError('Masked external truth cannot carry invented binding')
        return None
    if type(category) is not int or category < 0:
        raise ValueError('Existing external eligible semantic binding only')
    return kind, category


def conditional_diagnostics(head, grouped):
    if not grouped or not any(k[0] == 'Known' for k in grouped) or not any(k[0] == 'declared_pseudoNovel' for k in grouped):
        raise ValueError('Both independent eligible development roles required')
    # Table is ephemeral/private. Class macro is over the observed existing
    # semantic classes, not a balanced binary prior or deployment prevalence.
    table = sorted((score, float(label[0] == 'Known'), 1. / (len(grouped) * len(scores)))
                   for label, scores in grouped.items() for score in scores)
    private = np.asarray(table, dtype='<f8')
    result = membership_diagnostics(head, private)
    result['declared_pseudo_groups'] = result.pop('policy_pseudo_groups')
    result['observed_eligible_semantic_classes'] = len(grouped)
    result['independent_Train_dev_no_optimizer_or_selection'] = True
    result['declared_pseudo_is_existing_Known_derived_not_new_Novel_GT'] = True
    return result


def full_development_reliability(videos, labels, known, cfg, guard, *, head, initial):
    """ONE input-only current-prefix pass, full clock and all physical tracks.

    Group-level complete-observation and fixed observed-prefix track statistics
    are separate estimands. Repeated observations/prefixes are not independent
    replicates. Conflicting atomic aliases are explicitly masked, never split
    for convenient label-specific scores. No private score/target table output.
    """
    if head.training or initial.training or any(p.requires_grad for m in (head, initial) for p in m.parameters()):
        raise ValueError('Frozen endpoints and initial control, no fitting in development')
    if not videos or len({v for v, _ in videos}) != len(videos):
        raise ValueError('Unique full development videos, not repeated padding')
    prefixes = tuple(cfg['prefix_snapshots'])
    grouped = defaultdict(list); counts = Counter(); seen = Counter(); last = {}
    snapshots = {n: {} for n in prefixes}
    for video, arrays in videos:
        router = CurrentObservationRouter()
        for frame, rows in policy_current_frames(arrays):
            guard(); batch = router.ingest(frame, rows)
            counts['metadata_frames'] += 1; counts['empty_frames'] += int(not rows)
            counts['physical_observations'] += len(rows)
            for view, aliases in zip(batch.model_views, batch.routing_groups):
                # No external target/key is passed to the input-only kernel.
                cosine = max_known_cosine(view, known)
                counts['canonical_groups'] += 1
                target = external_group_target(labels, video, aliases)
                if target is None:
                    counts['masked_groups'] += 1
                else:
                    counts[target[0] + '_groups'] += 1; grouped[target].append(cosine)
                for route in aliases:
                    key = (video, route); seen[key] += 1; last[key] = (target, cosine)
                    if seen[key] in snapshots:
                        snapshots[seen[key]][key] = last[key]
    if (set(seen) != set(labels) or len(seen) != cfg['physical_tracks']
            or (counts['physical_observations'], counts['metadata_frames'], counts['empty_frames'])
            != (cfg['observations'], cfg['metadata_frames'], cfg['empty_frames'])):
        raise ValueError('Full physical denominator including empty, unknown and short tracks')
    def report(rows):
        if set(rows) != set(labels):
            raise ValueError('Every physical prefix retained')
        scoped = defaultdict(list); masked = 0
        for target, cosine in rows.values():
            if target is None: masked += 1
            else: scoped[target].append(cosine)
        return {'physical_tracks': len(rows), 'masked_tracks': masked,
                'initial_fixed_Raw_task_score': conditional_diagnostics(initial, scoped),
                'fixed_formal_endpoint_task_score': conditional_diagnostics(head, scoped)}
    result = {str(n): report({k: snapshots[n].get(k, last[k]) for k in labels}) for n in prefixes}
    result['all_available'] = report(last)
    guard()
    return {'full_clock': dict(counts), 'complete_group_observation_task': {
                'initial_fixed_Raw_task_score': conditional_diagnostics(initial, grouped),
                'fixed_formal_endpoint_task_score': conditional_diagnostics(head, grouped)},
            'fixed_observed_prefix_track_tasks': result,
            'all_physical_tracks_and_unknowns_retained': True,
            'unknown_conflicting_groups_as_negative_BG_Novel': False,
            'class_macro_or_role_conditional_task_is_deployment_prior': False,
            'Existing_gate_foreground_purity_or_full_posterior_calibrated': False,
            'bin_threshold_seed_model_checkpoint_or_control_selection': False,
            'private_score_target_tables_exported': False,
            'repeated_observations_prefixes_are_independent_replicates': False}
