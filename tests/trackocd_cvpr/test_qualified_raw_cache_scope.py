import numpy as np
import pytest
from src.trackocd_cvpr.qualified_raw_cache_scope import qualified_raw_cache_scope, original_inputs
from src.trackocd_cvpr import frame_atomic_state_ema as ema, versioned_reliability_memory as reliability
from src.trackocd_cvpr.M_development_rollout import readonly_M_stream
from src.trackocd_cvpr.M_development_probes import numeric_trace
from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
from tests.trackocd_cvpr.test_M_development_prerequisites import tiny_videos


@pytest.mark.parametrize('name', ['Mean_latest', 'StateEMA_latest', 'relative_quarantine', 'relative_weighted', 'pooled_weighted'])
def test_complete_synthetic_controller_shadow_preserves_events_versions_ledger_and_outcomes(name):
    videos = tiny_videos(); labels = {(1, '17'): ('declared_pseudoNovel', 20),
        (2, '17'): ('declared_pseudoNovel', 20), (3, '17'): ('Known', 10)}
    v = np.zeros(768, np.float32); v[7] = 1
    cfg = {'physical_tracks': 3, 'observations': 6, 'metadata_frames': 6, 'empty_frames': 0,
        'prefix_snapshots': [1, 2, 4, 8, 16], 'max_versions': 100,
        'raw_fixed_thresholds': {'tau_known': .65, 'tau_existing': .55}}
    model = None if name in ('Mean_latest', 'StateEMA_latest') else MemberReliabilityMLP(pooled_control=name == 'pooled_weighted').eval().requires_grad_(False)
    old, original = readonly_M_stream(videos, labels, {0: v}, {0: 10}, cfg, name, 1027, lambda: None, model=model)
    with qualified_raw_cache_scope() as cache:
        new, altered = readonly_M_stream(videos, labels, {0: v}, {0: 10}, cfg, name, 1027, lambda: None, model=model)
        assert cache.retained_bytes <= 67108864
    assert old == new and numeric_trace(original) == numeric_trace(altered)
    assert ema.committed_reliability_inputs is reliability.committed_reliability_inputs is original_inputs


def test_exception_restores_original_process_local_bindings_and_nesting_refused():
    with pytest.raises(RuntimeError):
        with qualified_raw_cache_scope():
            with pytest.raises(ValueError):
                with qualified_raw_cache_scope(): pass
            raise RuntimeError('synthetic interruption')
    assert ema.committed_reliability_inputs is reliability.committed_reliability_inputs is original_inputs
