import numpy as np
import pytest
from src.trackocd_cvpr.frame_atomic_state_ema import FrameAtomicStateEMA, StateEMAMemberControl
from src.trackocd_cvpr.prefix_member_bank import PrefixMemberBank
from src.trackocd_cvpr.causal_observation_router import CurrentObservationRouter
from src.trackocd_cvpr.version_lineage_audit import audit_version_lineage
from tests.trackocd_cvpr.test_versioned_reliability_memory import obs, reuse


def test_true_state_ema_equation_and_duplicate_refresh_not_double_update():
    bank, router, ema = PrefixMemberBank(), CurrentObservationRouter(), FrameAtomicStateEMA(momentum=.9)
    bank.apply_frame(router.ingest(0, [obs('a')]), ['S:0'])
    first = ema.refresh(bank)['S:0'].prototype(); saved = ema.versions()
    assert np.allclose(first[:2], [1, 0])
    bank.apply_frame(router.ingest(1, [obs('a', 1)]), ['S:0'])
    value = ema.refresh(bank)['S:0'].prototype()
    expected = .9 * np.eye(1, 768)[0] + .1 * bank.state_evidence()['S:0'].prototype()
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(value, expected, rtol=1e-7, atol=1e-7)
    current = ema.versions()
    np.testing.assert_array_equal(ema.refresh(bank)['S:0'].prototype(), value)
    assert ema.versions() == current and current[:len(saved)] == saved


def test_momentum_zero_is_latest_member_mean_not_append_cumulative_vote():
    bank, router, ema = PrefixMemberBank(), CurrentObservationRouter(), FrameAtomicStateEMA(momentum=0.)
    for frame in range(3):
        bank.apply_frame(router.ingest(frame, [obs('a', frame), obs('alias', frame)]), ['S:0'])
        state = ema.refresh(bank)['S:0']
        assert state.support_groups == 1 and state.routed_members == 2
        np.testing.assert_allclose(state.prototype(), bank.state_evidence()['S:0'].prototype(), atol=1e-7)


def test_same_frame_alias_peer_order_and_opaque_ID_rename_do_not_affect_EMA():
    left = StateEMAMemberControl({}, reuse); right = StateEMAMemberControl({}, reuse)
    for frame, a, b in ((0, [obs('x'), obs('y', 1)], [obs('999', 1), obs('renamed')]),
                        (1, [obs('x', 2)], [obs('renamed', 2)])):
        assert left.ingest(frame, a).events == right.ingest(frame, b).events
        assert left.memory_versions() == right.memory_versions()
    assert all(v.lifecycle != 'Confirmed' for v in left.memory_versions())


def test_same_frame_sees_no_peer_new_state_and_empty_refresh_no_latent_decay():
    calls = []
    def score(f, c): calls.append(len(f)); return reuse(f, c)
    control = StateEMAMemberControl({}, score)
    first = control.ingest(0, [obs('a'), obs('b', 1)])
    assert calls == [0, 0] and all(e.action == 'NEW' for e in first.events)
    saved = control.memory_versions(); control.ingest(1, [])
    assert control.memory_versions() == saved


def test_tombstone_transfer_parent_and_strict_known_pollution_cannot_be_reset():
    bank, router, ema = PrefixMemberBank(), CurrentObservationRouter(), FrameAtomicStateEMA()
    batch = router.ingest(0, [obs('known'), obs('pseudo', 1)])
    bank.apply_frame(batch, ['S:0' if v.visual[0, 0] else 'S:1' for v in batch.model_views]); ema.refresh(bank)
    bank.apply_frame(router.ingest(1, [obs('known', 2)]), ['S:1'])
    state = ema.refresh(bank)
    assert state['S:0'].prototype() is None
    audit = audit_version_lineage(ema.versions(), bank.commit_history(),
        {(4, 'known'): ('Known', 0), (4, 'pseudo'): ('policy_pseudo', 1)}, {0: 4})
    assert audit['strict_latest_version_status_counts']['VERIFIED_BAD_ANONYMOUS_KNOWN_HISTORY'] == 2
    old = ema.versions()
    bank.apply_frame(router.ingest(2, [obs('known', 3)]), ['S:0']); ema.refresh(bank)
    assert ema.versions()[:len(old)] == old  # revival retains old dirty source


def test_whole_frame_bound_failure_rolls_back_EMA_router_bank_and_births():
    control = StateEMAMemberControl({}, reuse, max_versions=1)
    with pytest.raises(ValueError, match='version bound'): control.ingest(0, [obs('a'), obs('b', 1)])
    bank, next_id = control.state_snapshot()
    assert bank.commit_history() == () and next_id == 0 and control.memory_versions() == ()
    control.ingest(0, [obs('a')])
    assert len(control.memory_versions()) == 1
