from copy import deepcopy
import pytest
from src.trackocd_cvpr.self_excluded_policy_rollout import (
    MODES, readonly_excluded_stream, actual_excluded_probes, excluded_raw_cache_scope, trace_numeric)
from src.trackocd_cvpr.M_development_rollout import readonly_M_stream
from tests.trackocd_cvpr.test_policy_training_data import arrays


def inputs():
    import numpy as np
    videos = [(1, arrays())]
    labels = {(1, '12'): ('Known', 4), (1, '99'): ('declared_pseudoNovel', 9)}
    known = {0: np.eye(1, 768, dtype=np.float32)[0]}
    cfg = {'physical_tracks': 2, 'observations': 3, 'metadata_frames': 4, 'empty_frames': 2,
        'raw_fixed_thresholds': {'tau_known': .65, 'tau_existing': .55},
        'prefix_snapshots': [1, 2, 4, 8, 16], 'max_versions': 65536}
    return videos, labels, known, {0: 4}, cfg


@pytest.mark.parametrize('mode', MODES)
def test_complete_numeric_cache_shadow_ID_future_and_filtered_view_bindings(mode):
    v, labels, known, bindings, cfg = inputs()
    old, old_trace = readonly_excluded_stream(v, labels, known, bindings, cfg, mode, 1027, lambda: None)
    with excluded_raw_cache_scope() as (raw, fingerprints):
        new, new_trace = readonly_excluded_stream(v, labels, known, bindings, cfg, mode, 1027, lambda: None)
        assert old == new and old_trace == new_trace
        p = actual_excluded_probes(v, labels, known, bindings, cfg, mode, 1027, lambda: None, new, new_trace)
        assert p['prior_events_versions_ledger_and_filtered_query_views_invariant']
        assert p['actual_future_observations_mutated'] > 0
        assert raw.retained_bytes + fingerprints.retained_bytes <= 67108864
    assert old['full_physical_bitset_scope'] == 2
    if mode != 'Mean_latest':
        assert old['actual_query_view_counts']['query_views'] == 3
        assert all(len(row.query_views) == len(row.result.events) for row in old_trace)


def test_baseline_is_exact_original_Mean_stream_not_a_new_unmatched_method():
    v, labels, known, bindings, cfg = inputs()
    base, trace = readonly_M_stream(v, labels, known, bindings, cfg, 'Mean_latest', 1027, lambda: None)
    new, new_trace = readonly_excluded_stream(v, labels, known, bindings, cfg, 'Mean_latest', 1027, lambda: None)
    for key in base:
        if key in new: assert base[key] == new[key]
    assert [(r.result.events, r.chosen_versions, r.version_history_sha256) for r in trace] == [
        (r.result.events, r.chosen_versions, r.version_history_sha256) for r in new_trace]


def test_cache_scope_exception_restores_extra_alias_and_refuses_unknown_nested_overlay():
    from src.trackocd_cvpr import self_excluded_mean_control as control
    from src.trackocd_cvpr.reliability_support_bridge import committed_reliability_inputs
    with pytest.raises(RuntimeError):
        with excluded_raw_cache_scope():
            with pytest.raises(ValueError):
                with excluded_raw_cache_scope(): pass
            raise RuntimeError('test')
    assert control.committed_reliability_inputs is committed_reliability_inputs


def test_missing_physical_frame_and_unknown_model_refuse_without_partial_acceptance():
    v, labels, known, bindings, cfg = inputs()
    with pytest.raises(ValueError): readonly_excluded_stream(v, labels, known, bindings, cfg, 'StateEMA_latest', 1027, lambda: None)
    bad = deepcopy(cfg); bad['observations'] += 1
    with pytest.raises(ValueError, match='Complete fixed'):
        readonly_excluded_stream(v, labels, known, bindings, bad, 'exclude_current_video', 1027, lambda: None)
