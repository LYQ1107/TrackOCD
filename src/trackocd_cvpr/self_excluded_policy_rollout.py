"""Full-clock Mean support-exclusion controls; labels are external audit only.

Private trace includes the ACTUAL query-aligned filtered Raw/numeric receipts.
Original chosen versions remain conservative dirty-history references, not
effective-support purity/independence certificates. No weights/assets loaded.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import numpy as np
import torch

from .C_development_rollout import raw_fixed_scores
from .frame_atomic_state_ema import StateEMAMemberControl
from .self_excluded_mean_control import SelfExcludedMeanMemberControl, QueryMeanViewReceipt
from .policy_training_data import policy_current_frames
from .versioned_causal_outcome_audit import VersionedCausalOutcomeAudit
from .qualified_fingerprint_cache_scope import qualified_fingerprint_cache_scope
from .reliability_support_bridge import committed_reliability_inputs as original_inputs

MODES = ('Mean_latest', 'exclude_query_alias_scope', 'exclude_current_video')


@contextmanager
def excluded_raw_cache_scope():
    """Own-process extra original-input alias; exact contract/finally restore.

Real complete old/new controller shadow is required by the registered worker;
synthetic tests or old unrelated controls do not establish this integration.
"""
    from . import self_excluded_mean_control as excluded
    from . import frame_atomic_state_ema as ema
    if excluded.committed_reliability_inputs is not original_inputs:
        raise ValueError('Original excluded Mean input binding required, no nesting')
    with qualified_fingerprint_cache_scope() as (raw, fingerprints):
        excluded.committed_reliability_inputs = ema.committed_reliability_inputs
        try: yield raw, fingerprints
        finally: excluded.committed_reliability_inputs = original_inputs


@dataclass(frozen=True)
class ExcludedFrameTrace:
    # PRIVATE ephemeral audit correspondence, never uploaded frame rows.
    video: int
    result: object
    chosen_versions: tuple
    version_history_sha256: str
    query_views: tuple


def trace_numeric(rows):
    return tuple((r.result.events, r.chosen_versions, r.version_history_sha256, r.query_views) for r in rows)


def readonly_excluded_stream(videos, labels, known, bindings, cfg, mode, order_seed, guard, *, rename_and_reverse=False):
    if mode not in MODES: raise ValueError('Only registered Mean exclusion controls, no EMA/M shortcut')
    order = np.random.default_rng(order_seed).permutation(len(videos)).tolist()
    routes = {}
    if rename_and_reverse:
        for video, arrays in videos:
            routes[video] = {int(t): 'excluded-renamed-' + str(len(arrays['track_id']) - i - 1)
                             for i, t in enumerate(arrays['track_id'])}
        labels = {(v, routes[v][int(t)]): label for (v, t), label in labels.items()}
    def score(f, c): return raw_fixed_scores(f, c, **cfg['raw_fixed_thresholds'])
    limits = {'max_members': cfg['physical_tracks'], 'max_tokens': cfg['observations'],
        'max_commits': cfg['observations'], 'max_payload_bytes': 67108864}
    common = dict(K=8, max_versions=cfg['max_versions'], bank_limits=limits)
    control = (StateEMAMemberControl(known, score, momentum=0., **common) if mode == 'Mean_latest'
        else SelfExcludedMeanMemberControl(known, score, exclusion_mode=mode, **common))
    audit = VersionedCausalOutcomeAudit(labels, bindings, prefixes=tuple(cfg['prefix_snapshots']),
        max_versions=cfg['max_versions'], max_commits=cfg['observations'])
    frames = empty = observations = consumed = 0
    actions, views_summary, private, ledger = Counter(), Counter(), [], hashlib.sha256()
    with torch.inference_mode():
        for position, ordinal in enumerate(order):
            video, arrays = videos[ordinal]
            if position: control.start_next_video()
            audit.start_video(position, video)
            for frame, rows in policy_current_frames(arrays, routing_bindings=routes.get(video)):
                if rename_and_reverse: rows = tuple(reversed(rows))
                guard(); result = control.ingest(frame, rows)
                views = () if mode == 'Mean_latest' else control.query_view_receipts()
                if mode != 'Mean_latest':
                    if len(views) != len(result.events) or any(type(v) is not QueryMeanViewReceipt or v.mode != mode for v in views):
                        raise ValueError('Every actual query must bind its filtered numerical view')
                    for v in views:
                        if (v.independent_semantic_purity_or_actual_development_accepted
                                or not v.original_version_reference_is_conservative_history_only
                                or not 0 <= v.retained_groups <= v.original_groups
                                or not 0 <= v.retained_routed_members <= v.original_routed_members):
                            raise ValueError('Real complete support exclusion counts, no purity/independence certificate')
                        views_summary.update({'query_views': 1, 'original_group_opportunities': v.original_groups,
                            'retained_group_opportunities': v.retained_groups,
                            'views_with_excluded_groups': int(v.retained_groups < v.original_groups),
                            'views_with_zero_retained_groups': int(not v.retained_groups)})
                bank, _ = control.state_snapshot(); versions = control.memory_versions()
                audit.audit_frame(video, result, control.chosen_version_bindings(), versions, bank.commit_history())
                for v in versions[consumed:]:
                    b = repr(v).encode('utf8'); ledger.update(len(b).to_bytes(8, 'little')); ledger.update(b)
                consumed = len(versions)
                private.append(ExcludedFrameTrace(video, result, control.chosen_version_bindings(), ledger.hexdigest(), views))
                frames += 1; empty += int(not rows); observations += len(rows)
                for aliases, event in zip(result.routing_groups, result.events): actions[event.action] += len(aliases)
    if (frames, empty, observations) != (cfg['metadata_frames'], cfg['empty_frames'], cfg['observations']):
        raise ValueError('Complete fixed physical clock, no short/unknown/empty pruning')
    return {**audit.finish(), 'order_seed': order_seed,
        'private_video_ordinal_order_sha256': hashlib.sha256(np.asarray(order, dtype='<i8').tobytes()).hexdigest(),
        'metadata_frames': frames, 'empty_frames': empty, 'observations': observations,
        'physical_actions': dict(actions), 'actual_query_view_counts': dict(views_summary),
        'optimizer_steps': 0, 'model_state_unchanged': True, 'GT_oracle_control_not_Pred': True,
        'conservative_original_history_not_filtered_active_purity_or_semantic_independence': True}, tuple(private)


def actual_excluded_probes(videos, labels, known, bindings, cfg, mode, seed, guard, first_result, first_trace):
    renamed, renamed_trace = readonly_excluded_stream(videos, labels, known, bindings, cfg, mode, seed, guard, rename_and_reverse=True)
    if renamed != first_result or trace_numeric(renamed_trace) != trace_numeric(first_trace):
        raise ValueError('Actual full ID/reverse changes event/chosen version/numeric ledger/filtered query view or audit')
    ordinal = int(np.random.default_rng(seed).permutation(len(videos))[0]); video, arrays = videos[ordinal]
    if not len(arrays['frame_index']): raise ValueError('Nonvacuous observed first-video prefix required')
    split = int(arrays['frame_index'].min()); future = arrays['frame_index'] > split
    if not future.any(): raise ValueError('Actual future unavailable, never a vacuous causal pass')
    changed = {key: value.copy() for key, value in arrays.items()}
    changed['visual'][future] = 0; changed['visual'][future, 0] = 1
    count = int(np.any(changed['visual'][future] != arrays['visual'][future], axis=1).sum())
    if not count: raise ValueError('Actual future mutation cannot be a numeric no-op')
    altered = list(videos); altered[ordinal] = (video, changed)
    _, altered_trace = readonly_excluded_stream(altered, labels, known, bindings, cfg, mode, seed, guard)
    left = tuple(row for row in first_trace if row.video == video and row.result.frame_index <= split)
    right = tuple(row for row in altered_trace if row.video == video and row.result.frame_index <= split)
    if not left or not any(row.result.events for row in left) or trace_numeric(left) != trace_numeric(right):
        raise ValueError('Earlier prediction/chosen version/full ledger/filtered view changed from future mutation')
    return {'actual_full_ID_rename_reverse_event_ledger_query_view_and_audit_equal': True,
        'actual_future_observations_mutated': count,
        'prior_events_versions_ledger_and_filtered_query_views_invariant': True,
        'future_scope_first_video_only_not_exhaustive_all_video_causal_proof': True}
