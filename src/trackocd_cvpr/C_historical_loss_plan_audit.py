"""Read-only replay of ACTUAL recorded C actions, external loss plans only.

Caller must gate the full original cohort/source/authority/private history hash
and qualified fit/policy caches first. Reconstruct numeric candidates from Raw,
then apply recorded actions (never GT-selected actions). Compare ALL episode
summaries, actual update-calendar counters and recorded anonymous label history.
No model/logits/gradients recovered; no new inference, fitting or memory repair.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import json
import numpy as np

from .atomic_member_control import MemberFrameResult
from .candidate_score_contract import build_candidates, unit
from .candidate_training_loss import ExternalPredictionHistory, lawful_label, prediction_owned_loss_plan
from .causal_observation_router import CurrentObservationRouter
from .policy_training_data import policy_current_frames
from .prefix_member_bank import PrefixMemberBank


@dataclass(frozen=True)
class RecordedCAction:
    action: str
    category_id: int | None
    token: str | None
    ambiguity_reason: str | None


def recorded_action(record, bindings, bank, next_id):
    if set(record) != {'action', 'category_id', 'token', 'ambiguity_reason'}:
        raise ValueError('Exact original recorded action schema; no fabricated logits')
    event = RecordedCAction(**record)
    if event.action not in ('KNOWN', 'EXISTING', 'NEW', 'WAIT'):
        raise ValueError('Actual original output action required')
    if event.action in ('KNOWN', 'EXISTING'):
        target = event.category_id if event.action == 'KNOWN' else event.token
        if (event.ambiguity_reason is not None
                or (event.action == 'KNOWN' and (type(target) is not int or event.token is not None))
                or (event.action == 'EXISTING' and (type(target) is not str or event.category_id is not None))
                or not any(len(group) == 1 and group[0].kind == event.action
                           and group[0].identifier == target for group in bindings)):
            raise ValueError('Actual resolvable exposed output binding required')
    elif event.action == 'NEW':
        if (event.token != f'S:{next_id}' or event.category_id is not None
                or event.ambiguity_reason is not None or event.token in bank._tokens):
            raise ValueError('Actual canonical sequential prediction-owned birth required')
        next_id += 1
    elif (event.token is not None or event.category_id is not None
            or event.ambiguity_reason not in (None, 'AMBIGUOUS_MAXIMA', 'INDISTINGUISHABLE_BINDINGS')):
        raise ValueError('Actual recorded WAIT output required')
    return event, next_id


def positive_kinds(plan, bindings):
    kinds = set()
    for i, positive in enumerate(plan.positive):
        if not positive: continue
        if i == len(bindings): kinds.add('NEW')
        elif i == len(bindings) + 1: kinds.add('WAIT')
        else:
            if len(bindings[i]) != 1: raise ValueError('Resolvable original supervised row required')
            kinds.add(bindings[i][0].kind)
    return tuple(sorted(kinds))


def audit_recorded_C_trial(videos, targets, known, window, variant, seed, private_episodes, summary, guard):
    """No assets or model calls here; every cached frame and recorded row kept.

Positive group counts indicate loss plans at reconstructed update frames,
NOT gradients, a semantic-independent sample count or a causal collapse proof.
Actual source/window and all existing calendar/summary checks are mandatory.
"""
    settings, training = window['variants'][variant], window['training']
    if (len(private_episodes) != training['episodes'] or len(summary['episodes']) != training['episodes']
            or len(summary['loss_curve']) != training['optimizer_steps']):
        raise ValueError('Complete original episodes and actual optimizer calendar required')
    known = {key: unit(value) for key, value in known.items()}
    rng = np.random.default_rng(seed); curve_cursor = 0
    overall_families = Counter(); update_labels = set(); pseudo_labels = set()
    overall_positive = Counter(); overall_reasons = Counter(); reports = []
    for episode_index, saved in enumerate(private_episodes):
        guard()
        order = rng.permutation(len(videos)).tolist()
        if (set(saved) != {'episode', 'video_ordinal_order', 'anonymous_commit_labels', 'actual_predictions'}
                or saved['episode'] != episode_index or saved['video_ordinal_order'] != order):
            raise ValueError('Actual registered seed/order/full episode provenance required')
        bank, history, next_id = PrefixMemberBank(), ExternalPredictionHistory(), 0
        frames = observations = empty = groups = steps = cursor = 0
        reasons, actions, all_families, update_positive, update_reasons = Counter(), Counter(), Counter(), Counter(), Counter()
        other_all = cross_all = 0
        for position, ordinal in enumerate(order):
            video, arrays = videos[ordinal]
            if position: bank.start_next_video()
            router = CurrentObservationRouter()
            for frame, rows in policy_current_frames(arrays):
                guard()
                if cursor >= len(saved['actual_predictions']): raise ValueError('Missing actual clock frame')
                recorded = saved['actual_predictions'][cursor]; cursor += 1
                batch = router.ingest(frame, rows)
                aliases = tuple(tuple(group) for group in recorded['routing_groups'])
                if (set(recorded) != {'video_id', 'frame_index', 'routing_groups', 'events'}
                        or recorded['video_id'] != video or recorded['frame_index'] != frame
                        or aliases != batch.routing_groups or len(recorded['events']) != len(batch.model_views)):
                    raise ValueError('Complete current Raw/clock/actual recorded canonical alias alignment')
                states, prior = bank.state_evidence(), history.snapshot()
                events, assignments, terms = [], [], set()
                frame_families, frame_positive, frame_reasons = Counter(), Counter(), Counter()
                masked = other = cross = 0
                for view, routes, raw_event in zip(batch.model_views, aliases, recorded['events']):
                    candidates = build_candidates(view, known, states, K=settings['K_per_family'])
                    event, next_id = recorded_action(raw_event, candidates.bindings, bank, next_id)
                    events.append(event); assignments.append(event.token if event.action in ('NEW', 'EXISTING') else None)
                    keys = [(video, route) for route in routes]
                    plan = prediction_owned_loss_plan(candidates.bindings, keys, targets, prior)
                    reasons[plan.reason] += 1
                    if not any(plan.positive):
                        masked += 1
                    else:
                        label = lawful_label(targets[keys[0]])
                        terms.add(label); frame_families[plan.query_family] += 1
                        frame_reasons[plan.query_family + ':' + plan.reason] += 1
                        for kind in positive_kinds(plan, candidates.bindings):
                            frame_positive[plan.query_family + ':' + kind] += 1
                        other += int(plan.positive_has_other_physical_member)
                        cross += int(plan.positive_has_other_video_member)
                    actions[event.action] += len(routes)
                # Only actual recorded model outputs change Raw bank. Labels
                # entered external plan above, never assignment or prototype.
                bank.apply_frame(batch, assignments)
                result = MemberFrameResult(frame, tuple(events), aliases)
                all_families.update(frame_families); other_all += other; cross_all += cross
                if terms and steps < training['updates_per_episode']:
                    if curve_cursor >= len(summary['loss_curve']): raise ValueError('Missing actual update receipt')
                    row = summary['loss_curve'][curve_cursor]
                    expected = {'step': curve_cursor + 1, 'episode': episode_index, 'episode_optimizer_step': steps + 1,
                        'actual_optimizer_family_groups': dict(frame_families), 'supervised_classes_this_update': len(terms),
                        'masked_groups_this_frame': masked, 'positive_other_physical_groups_this_update': other,
                        'positive_other_video_groups_this_update': cross}
                    if any(row[key] != value for key, value in expected.items()):
                        raise ValueError('Every reconstructed update calendar/exposure must match actual recorded fit')
                    if not np.isfinite(row['loss']) or not np.isfinite(row['gradient_norm_before_clip']) or row['gradient_norm_before_clip'] <= 0:
                        raise ValueError('Actual finite nonzero optimizer receipt, no invented update')
                    steps += 1; curve_cursor += 1
                    overall_families.update(frame_families); update_labels.update(terms)
                    pseudo_labels.update(label for label in terms if label[0] == 'policy_pseudo')
                    update_positive.update(frame_positive); update_reasons.update(frame_reasons)
                history.observe_frame(video, result, targets)
                frames += 1; empty += int(not rows); observations += len(rows); groups += len(events)
        expected = {'episode': episode_index, 'optimizer_steps': steps, 'videos': len(videos),
            'metadata_frames': frames, 'empty_frames': empty, 'physical_observations': observations,
            'canonical_groups': groups, 'physical_actions': dict(actions), 'loss_plan_counts_all_frames': dict(reasons),
            'eligible_family_groups_all_frames_including_frozen_tail': dict(all_families),
            'positive_other_physical_groups_all_frames': other_all, 'positive_other_video_groups_all_frames': cross_all,
            'actual_anonymous_commits_retained': len(history.records()), 'actual_bank_commits_retained': len(bank.commit_history()),
            'allocated_tokens': next_id}
        if (steps != training['updates_per_episode'] or cursor != len(saved['actual_predictions'])
                or summary['episodes'][episode_index] != expected
                or json.loads(json.dumps([asdict(row) for row in history.records()])) != saved['anonymous_commit_labels']):
            raise ValueError('Complete episode summary/actual retained private history must match, no subset')
        overall_positive.update(update_positive); overall_reasons.update(update_reasons)
        reports.append({'episode': episode_index, 'optimizer_frames_matched': steps,
            'metadata_frames_matched': frames, 'physical_observations_matched': observations,
            'update_positive_plan_groups_by_family_and_kind': dict(update_positive),
            'update_plan_groups_by_family_and_reason': dict(update_reasons)})
    if (curve_cursor != training['optimizer_steps'] or dict(overall_families) != summary['actual_optimizer_family_groups']
            or len(update_labels) != summary['actual_optimizer_semantic_classes']
            or len(pseudo_labels) != summary['actual_optimizer_policy_pseudo_classes']):
        raise ValueError('All actual 200-update family/class exposure counters required')
    return {'all_original_episode_summaries_update_calendars_and_retained_histories_match': True,
        'episodes': reports, 'historical_optimizer_frames_matched': curve_cursor,
        'update_positive_plan_groups_by_family_and_kind': dict(overall_positive),
        'update_plan_groups_by_family_and_reason': dict(overall_reasons),
        'historical_logits_gradients_or_rowwise_numeric_trace_recovered': False,
        'new_optimizer_steps': 0, 'actual_model_inference_or_GT_memory_repair': False,
        'positive_groups_are_loss_plan_opportunities_not_independent_samples_or_gradient_magnitudes': True,
        'collapse_cause_new_training_development_or_Goal_accepted': False}
