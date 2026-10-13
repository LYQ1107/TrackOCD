import numpy as np
import pytest
from src.trackocd_cvpr.effective_mean_observer import observed_mean_constructors
from src.trackocd_cvpr.self_excluded_policy_rollout import readonly_excluded_stream, MODES
from tests.trackocd_cvpr.test_self_excluded_policy_rollout import inputs


@pytest.mark.parametrize('mode', MODES)
def test_actual_synthetic_sealed_predictions_history_and_views_unchanged_by_external_observer(mode):
    videos, labels, known, bindings, cfg = inputs()
    result, trace = readonly_excluded_stream(videos, labels, known, bindings, cfg, mode, 1027, lambda: None)
    order = [videos[i][0] for i in np.random.default_rng(1027).permutation(len(videos))]
    with observed_mean_constructors(labels, bindings, cfg['prefix_snapshots'], order) as observer:
        checked, checked_trace = readonly_excluded_stream(videos, labels, known, bindings, cfg, mode, 1027, lambda: None)
        active, active_trace = observer.finish()
    assert result == checked and trace == checked_trace
    assert active['full_physical_scope'] == 2 and active['metadata_frames_observed'] == 4
    assert active['physical_observations_observed'] == 3 and len(active_trace) == 4


def test_ID_reverse_preserves_effective_metrics_and_external_trace_not_routes():
    videos, labels, known, bindings, cfg = inputs(); outcomes = []
    order = [videos[i][0] for i in np.random.default_rng(1027).permutation(len(videos))]
    for rename in (False, True):
        with observed_mean_constructors(labels, bindings, cfg['prefix_snapshots'], order, rename=rename, videos=videos) as observer:
            readonly_excluded_stream(videos, labels, known, bindings, cfg, 'exclude_query_alias_scope', 1027, lambda: None, rename_and_reverse=rename)
            outcomes.append(observer.finish())
    assert outcomes[0] == outcomes[1]


def test_exception_restores_original_constructors_nested_observer_refused():
    from src.trackocd_cvpr import self_excluded_policy_rollout as rollout
    videos, labels, known, bindings, cfg = inputs(); old = rollout.StateEMAMemberControl, rollout.SelfExcludedMeanMemberControl
    with pytest.raises(RuntimeError):
        with observed_mean_constructors(labels, bindings, cfg['prefix_snapshots'], [1]):
            with pytest.raises(ValueError):
                with observed_mean_constructors(labels, bindings, cfg['prefix_snapshots'], [1]): pass
            raise RuntimeError('test')
    assert (rollout.StateEMAMemberControl, rollout.SelfExcludedMeanMemberControl) == old
