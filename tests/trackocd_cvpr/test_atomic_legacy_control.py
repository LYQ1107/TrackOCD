"""Synthetic executed mutations; not real checkpoint/dense benchmark proof."""
import numpy as np
import pytest

from src.trackocd_cvpr.atomic_legacy_control import AtomicLegacyControl
from src.trackocd_cvpr.causal_observation_router import CurrentObservation


def observation(identity, axis=0):
    return CurrentObservation(identity, np.eye(1, 768, axis, dtype=np.float32)[0],
                              np.array([0.1, 0.2, 0.5, 0.8]), 0.7)


def state_bytes(control):
    memory = control.state_snapshot()
    return (memory.next_id, tuple((key, value.vector_sum.tobytes(), value.count,
                                  value.quality_sum) for key, value in memory.anonymous.items()))


def test_exact_ties_write_once_then_broadcast_same_token():
    calls = []
    def policy(cues):
        calls.append(cues.tobytes())
        assert cues.shape == (17,)
        with pytest.raises(ValueError):
            cues.flags.writeable = True
        return 2
    control = AtomicLegacyControl({}, policy)
    result = control.ingest(0, [observation('9'), observation('2')])
    assert len(calls) == len(result.events) == 1
    assert [event.token for _, event in result.routed_predictions()] == ['S:0', 'S:0']
    assert control.state_snapshot().anonymous['S:0'].count == 1


def test_renumber_and_within_frame_reorder_preserve_actual_actions_cues_memory():
    def policy(cues):
        return 1 if cues[16] else 2
    left, right = AtomicLegacyControl({}, policy), AtomicLegacyControl({}, policy)
    a = left.ingest(0, [observation('1', 2), observation('99', 0)])
    b = right.ingest(0, [observation('opaque', 0), observation('zero', 2)])
    assert a.events == b.events
    a = left.ingest(2, [observation('99', 1), observation('1', 3)])
    b = right.ingest(2, [observation('zero', 3), observation('opaque', 1)])
    assert a.events == b.events
    assert state_bytes(left) == state_bytes(right)


def test_future_append_changes_only_future_outputs_and_retains_prior_result():
    left = AtomicLegacyControl({}, lambda cues: 2)
    right = AtomicLegacyControl({}, lambda cues: 2)
    prior = left.ingest(0, [observation('x')])
    assert prior.events == right.ingest(0, [observation('y')]).events
    retained = prior.events
    left.ingest(1, [observation('x', 1)])
    right.ingest(1, [observation('y', 4), observation('future-only', 6)])
    assert prior.events == retained


@pytest.mark.parametrize('bad', ('exception', 'wrong_type', 'unavailable_known'))
def test_later_group_failure_rolls_back_entire_frame_then_retry(bad):
    calls = []
    def policy(cues):
        calls.append(1)
        if len(calls) == 2:
            if bad == 'exception':
                raise RuntimeError('synthetic policy failure')
            return 2.0 if bad == 'wrong_type' else 0
        return 2
    control = AtomicLegacyControl({}, policy)
    before = state_bytes(control)
    with pytest.raises((ValueError, RuntimeError)):
        control.ingest(0, [observation('a', 1), observation('b', 2)])
    assert state_bytes(control) == before
    # Callback calls are external and intentionally NOT rolled back.
    result = control.ingest(0, [observation('a', 1), observation('b', 2)])
    assert len(result.events) == control.state_snapshot().next_id == 2


def test_empty_frame_advances_chronology_without_decision_or_memory_write():
    control = AtomicLegacyControl({}, lambda cues: 2)
    before = state_bytes(control)
    assert not control.ingest(0, []).events
    assert state_bytes(control) == before
    with pytest.raises(ValueError):
        control.ingest(0, [observation('a')])


def test_cross_video_memory_persists_without_recycling_tokens_or_physical_ids():
    control = AtomicLegacyControl({}, lambda cues: 2)
    assert control.ingest(4, [observation('same-id')]).events[0].token == 'S:0'
    control.start_next_video()
    second = control.ingest(0, [observation('same-id', 1)])
    assert second.events[0].token == 'S:1'
    assert control.state_snapshot().anonymous['S:1'].count == 1


def test_observed_state_bound_refuses_frame_without_eviction_or_retry_drift():
    control = AtomicLegacyControl({}, lambda cues: 2, max_anonymous=1)
    control.ingest(0, [observation('a')])
    before = state_bytes(control)
    with pytest.raises(ValueError):
        control.ingest(1, [observation('b', 1)])
    assert state_bytes(control) == before
    assert not control.ingest(1, []).events


def test_detached_snapshot_and_known_input_do_not_alias_runtime_state():
    prototype = observation('known', 2).visual
    control = AtomicLegacyControl({7: prototype}, lambda cues: 0)
    prototype[:] = 99
    snapshot = control.state_snapshot()
    snapshot.known[7][:] = 0
    result = control.ingest(0, [observation('a', 2)])
    assert result.events[0].category_id == 7
    assert np.argmax(control.state_snapshot().known[7]) == 2


def test_raw_cues_use_visible_normalized_uncertainty_and_effective_maturity():
    received = []
    def policy(cues):
        received.append(cues.copy())
        return 3
    control = AtomicLegacyControl({}, policy)
    first = observation('a', 1)
    first.visual[:] *= 50
    control.ingest(0, [first])
    control.ingest(1, [observation('a', 2)])
    assert received[0][7] == 0
    assert 0 < received[1][7] < 1
    assert received[1][8] == pytest.approx(np.log1p(2) / 4)


def test_unlabelled_interface_and_invalid_observation_preserve_memory():
    control = AtomicLegacyControl({}, lambda cues: 2)
    item = observation('a')
    item.visual[:] = 0
    with pytest.raises(ValueError):
        control.ingest(0, [item])
    assert state_bytes(control) == (0, ())
    assert control.ingest(0, [observation('a')]).events[0].token == 'S:0'
    with pytest.raises(TypeError):
        control.ingest(1, [observation('a')], category_id=4)
