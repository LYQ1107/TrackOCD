"""Actual frozen-M/Mean/EMA full replays; no no-op or empty future pass."""
import numpy as np
from .M_development_rollout import readonly_M_stream


def numeric_trace(private):
    # Routing aliases are bookkeeping, excluded only for the ID bijection.
    return tuple((row.result.events, row.chosen_versions, row.version_history_sha256) for row in private)


def actual_M_probes(videos, labels, known, bindings, cfg, name, seed, guard,
                    measured_first, private_first, *, model=None):
    renamed, renamed_private = readonly_M_stream(videos, labels, known, bindings, cfg,
        name, seed, guard, model=model, rename_and_reverse=True)
    if renamed != measured_first or numeric_trace(renamed_private) != numeric_trace(private_first):
        raise ValueError('Actual full M IDs/delivery affect policy, chosen version, numeric ledger or audit')
    ordinal = int(np.random.default_rng(seed).permutation(len(videos))[0])
    video, arrays = videos[ordinal]
    if not len(arrays['frame_index']):
        raise ValueError('First video has no observed prefix: future probe unavailable')
    split = int(arrays['frame_index'].min()); future = arrays['frame_index'] > split
    if not future.any():
        raise ValueError('No actual future observations: invariance is unavailable, not passed')
    changed = {key: value.copy() for key, value in arrays.items()}
    changed['visual'][future] = 0; changed['visual'][future, 0] = 1
    actual_changes = int(np.any(changed['visual'][future] != arrays['visual'][future], axis=1).sum())
    if not actual_changes:
        raise ValueError('Future mutation is a numeric no-op: unavailable, not passed')
    altered = list(videos); altered[ordinal] = (video, changed)
    _, altered_private = readonly_M_stream(altered, labels, known, bindings, cfg, name, seed, guard, model=model)
    left = tuple(row for row in private_first if row.video == video and row.result.frame_index <= split)
    right = tuple(row for row in altered_private if row.video == video and row.result.frame_index <= split)
    if not left or not any(row.result.events for row in left) or numeric_trace(left) != numeric_trace(right):
        raise ValueError('Nonvacuous earlier M predictions/chosen versions/numeric memory history changed')
    return {'actual_full_ID_rename_reverse_stream_equal': True,
            'actual_future_probe': {'future_observations_mutated': actual_changes,
                'verified_same_prior_events_chosen_versions_and_numeric_ledger': True},
            'future_probe_scope': 'First-video suffix mutation; not all-video exhaustive causal proof'}
