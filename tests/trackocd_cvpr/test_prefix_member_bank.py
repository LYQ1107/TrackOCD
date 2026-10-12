"""Executed synthetic evidence/state mutations; no real GT, caches or fitting."""
from dataclasses import replace
import numpy as np
import pytest
from src.trackocd_cvpr.causal_observation_router import CurrentObservation, CurrentObservationRouter
from src.trackocd_cvpr.causal_observation_router import visible_bytes
from src.trackocd_cvpr.prefix_member_bank import PrefixMemberBank


def obs(identity, axis=0, sign=1):
    vector = np.zeros(768, dtype=np.float32)
    vector[axis] = sign
    return CurrentObservation(identity, vector, np.array([.1, .1, .5, .5]), .8)


def test_single_member_replaces_prefix_instead_of_repeated_aggregate_writes():
    router, bank = CurrentObservationRouter(), PrefixMemberBank()
    bank.apply_frame(router.ingest(0, [obs('x')]), ['S:0'])
    batch = router.ingest(1, [obs('x', 1)])
    bank.apply_frame(batch, ['S:0'])
    state = bank.state_evidence()['S:0']
    assert state.support_groups == state.routed_members == 1
    assert np.allclose(state.prototype(), batch.model_views[0].weighted_mean())
    # Actual old repeated-prefix sum would tilt toward the first observation.
    legacy = obs('x').visual + batch.model_views[0].weighted_mean()
    assert not np.allclose(state.prototype(), legacy / np.linalg.norm(legacy))
    assert len(bank.commit_history()) == 2


def test_exact_aliases_share_support_then_divergence_splits_prospectively():
    router, bank = CurrentObservationRouter(), PrefixMemberBank()
    first = router.ingest(0, [obs('a'), obs('b')])
    bank.apply_frame(first, ['S:0'])
    assert bank.state_evidence()['S:0'].support_groups == 1
    assert bank.state_evidence()['S:0'].routed_members == 2
    sealed = bank.commit_history()
    later = router.ingest(1, [obs('a', 1), obs('b', 2)])
    bank.apply_frame(later, ['S:0'] * len(later.model_views))
    assert bank.state_evidence()['S:0'].support_groups == 2
    assert bank.commit_history()[:len(sealed)] == sealed


def test_partial_alias_advance_preserves_other_alias_committed_support():
    router, bank = CurrentObservationRouter(), PrefixMemberBank()
    bank.apply_frame(router.ingest(0, [obs('a'), obs('b')]), ['S:0'])
    bank.apply_frame(router.ingest(1, [obs('a', 1)]), ['S:0'])
    state = bank.state_evidence()['S:0']
    assert state.routed_members == state.support_groups == 2
    expected = obs('b').visual + np.asarray([1, 1] + [0] * 766) / np.sqrt(2)
    assert np.allclose(state.prototype(), expected / np.linalg.norm(expected))


def test_actual_route_renumber_reorder_preserve_state_features_not_route_ledger():
    left, right = PrefixMemberBank(), PrefixMemberBank()
    lr, rr = CurrentObservationRouter(), CurrentObservationRouter()
    for frame, a, b in ((0, [obs('9'), obs('2')], [obs('opaque-b'), obs('opaque-a')]),
                         (1, [obs('9', 1), obs('2', 2)], [obs('opaque-b', 2), obs('opaque-a', 1)])):
        lb, rb = lr.ingest(frame, a), rr.ingest(frame, b)
        left.apply_frame(lb, ['S:0'] * len(lb.model_views))
        right.apply_frame(rb, ['S:0'] * len(rb.model_views))
        assert left.state_evidence() == right.state_evidence()


def test_future_change_cannot_modify_sealed_history_or_earlier_evidence_snapshot():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('x')]), ['S:0'])
    history, snapshot = bank.commit_history(), bank.state_evidence()
    bank.apply_frame(router.ingest(1, [obs('x', 3), obs('future', 4)]), ['S:0', 'S:0'])
    assert bank.commit_history()[:1] == history
    assert snapshot['S:0'].prototype()[0] == 1


def test_assignment_revision_leaves_old_token_tombstone_and_past_history_intact():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('x')]), ['S:0'])
    history = bank.commit_history()
    bank.apply_frame(router.ingest(1, [obs('x', 1)]), ['S:1'])
    assert bank.state_evidence()['S:0'].status == 'EMPTY'
    assert bank.state_evidence()['S:1'].support_groups == 1
    assert bank.commit_history()[:1] == history
    assert bank.commit_history()[-1].previous_token == 'S:0'


def test_wait_and_empty_frame_preserve_prior_committed_support():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('x')]), ['S:0'])
    before, history = bank.state_evidence(), bank.commit_history()
    bank.apply_frame(router.ingest(1, [obs('x', 2)]), [None])
    bank.apply_frame(router.ingest(2, []), [])
    assert bank.state_evidence() == before and bank.commit_history() == history
    with pytest.raises(ValueError, match='chronology'):
        bank.apply_frame(router.ingest(3, []).__class__(2, (), (), ()), [])


def test_reused_physical_handle_in_next_video_is_new_support_not_replacement():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('x')]), ['S:0'])
    bank.start_next_video()
    router = CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('x')]), ['S:0'])
    assert bank.state_evidence()['S:0'].support_groups == 2
    assert [item.video_epoch for item in bank.commit_history()] == [0, 1]


def test_antipodal_distinct_members_keep_evidence_and_zero_resultant_not_drop():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    batch = router.ingest(0, [obs('x'), obs('y', sign=-1)])
    bank.apply_frame(batch, ['S:0', 'S:0'])
    state = bank.state_evidence()['S:0']
    assert state.status == 'ZERO_RESULTANT' and state.prototype() is None
    assert state.support_groups == 2 and state.dispersion == 1


@pytest.mark.parametrize('limit', ('members', 'tokens', 'commits', 'payload'))
def test_observed_bounds_fail_whole_frame_without_eviction_or_partial_history(limit):
    kwargs = {'max_' + limit: 1} if limit != 'payload' else {'max_payload_bytes': 1}
    bank, router = PrefixMemberBank(**kwargs), CurrentObservationRouter()
    batch = router.ingest(0, [obs('x'), obs('y', 1)])
    before = bank.state_evidence(), bank.commit_history()
    with pytest.raises(ValueError, match='bound'):
        bank.apply_frame(batch, ['S:0', 'S:1'])
    assert (bank.state_evidence(), bank.commit_history()) == before


def test_later_bad_token_is_atomic_and_same_frame_valid_retry_succeeds():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    batch = router.ingest(0, [obs('x'), obs('y', 1)])
    with pytest.raises(ValueError):
        bank.apply_frame(batch, ['S:0', 'GT:1'])
    assert bank.commit_history() == () and bank.state_evidence() == {}
    bank.apply_frame(batch, ['S:0', 'S:1'])
    assert len(bank.commit_history()) == 2


def test_mutated_past_prefix_and_fingerprint_cannot_replace_committed_member():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('x')]), ['S:0'])
    bad_router = CurrentObservationRouter()
    bad_router.ingest(0, [obs('x', 5)])
    changed_past = bad_router.ingest(1, [obs('x', 1)])
    before = bank.state_evidence(), bank.commit_history()
    with pytest.raises(ValueError, match='past'):
        bank.apply_frame(changed_past, ['S:0'])
    correct = router.ingest(1, [obs('x', 1)])
    with pytest.raises(ValueError, match='fingerprint'):
        bank.apply_frame(replace(correct, visible_input_sha256=('a' * 64,)), ['S:0'])
    assert (bank.state_evidence(), bank.commit_history()) == before
    bank.apply_frame(correct, ['S:0'])


def test_repeated_feature_at_distinct_times_not_deduplicated_as_one_alias():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('early')]), ['S:0'])
    bank.apply_frame(router.ingest(2, [obs('later')]), ['S:0'])
    assert bank.state_evidence()['S:0'].support_groups == 2


def test_no_GT_parameter_or_writable_prototype_alias_is_available():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    batch = router.ingest(0, [obs('x')])
    with pytest.raises(TypeError):
        bank.apply_frame(batch, ['S:0'], category_id=8)
    bank.apply_frame(batch, ['S:0'])
    value = bank.state_evidence()['S:0'].prototype()
    value[:] = 77
    assert bank.state_evidence()['S:0'].prototype()[0] == 1


def test_duplicate_visible_group_cannot_split_aliases_into_conflicting_tokens():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    batch = router.ingest(0, [obs('x')])
    view, signature = batch.model_views[0], batch.visible_input_sha256[0]
    bad = replace(batch, model_views=(view, view), routing_groups=(('x',), ('y',)),
                  visible_input_sha256=(signature, signature))
    with pytest.raises(ValueError, match='unique visible'):
        bank.apply_frame(bad, ['S:0', 'S:1'])
    assert bank.commit_history() == () and bank.state_evidence() == {}


@pytest.mark.parametrize('elapsed', (np.array([0.1]), np.array([1], dtype=np.int64)))
def test_malformed_elapsed_not_silently_cast_or_rebased_to_hide_future(elapsed):
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    batch = router.ingest(0, [obs('x')])
    view = replace(batch.model_views[0], elapsed_frames=elapsed)
    import hashlib
    bad = replace(batch, model_views=(view,), visible_input_sha256=(hashlib.sha256(visible_bytes(view)).hexdigest(),))
    with pytest.raises(ValueError, match='chronology'):
        bank.apply_frame(bad, ['S:0'])
    assert bank.commit_history() == ()


def test_video_bound_does_not_reset_state_or_recycle_tokens():
    bank = PrefixMemberBank(max_videos=1)
    router = CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('x')]), ['S:0'])
    before = bank.state_evidence(), bank.commit_history()
    with pytest.raises(ValueError, match='Video bound'):
        bank.start_next_video()
    assert (bank.state_evidence(), bank.commit_history()) == before


def test_equal_payloads_are_interned_but_cross_video_support_stays_distinct():
    bank = PrefixMemberBank()
    bank.apply_frame(CurrentObservationRouter().ingest(0, [obs('x')]), ['S:0'])
    bank.start_next_video()
    bank.apply_frame(CurrentObservationRouter().ingest(0, [obs('x')]), ['S:0'])
    supports = list(bank._members.values())  # Synthetic resource bookkeeping only.
    assert supports[0].payload is supports[1].payload
    assert bank.state_evidence()['S:0'].support_groups == 2


def test_wait_does_not_allow_an_input_to_rewrite_already_committed_past():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('x')]), ['S:0'])
    other = CurrentObservationRouter()
    other.ingest(0, [obs('x', 3)])
    before = bank.state_evidence(), bank.commit_history()
    with pytest.raises(ValueError, match='past'):
        bank.apply_frame(other.ingest(1, [obs('x', 4)]), [None])
    assert (bank.state_evidence(), bank.commit_history()) == before


def test_two_actual_streams_with_changed_future_have_equal_sealed_prefix_results():
    left, right = PrefixMemberBank(), PrefixMemberBank()
    lr, rr = CurrentObservationRouter(), CurrentObservationRouter()
    for bank, router in ((left, lr), (right, rr)):
        bank.apply_frame(router.ingest(0, [obs('x')]), ['S:0'])
    left_prefix, right_prefix = left.state_evidence(), right.state_evidence()
    sealed_left, sealed_right = left.commit_history(), right.commit_history()
    assert left_prefix == right_prefix and sealed_left == sealed_right
    left.apply_frame(lr.ingest(1, [obs('x', 1)]), ['S:0'])
    changed = rr.ingest(1, [obs('x', 3), obs('future', 4)])
    right.apply_frame(changed, ['S:0'] * len(changed.model_views))
    assert left.state_evidence() != right.state_evidence()
    assert left_prefix == right_prefix
    assert left.commit_history()[:1] == right.commit_history()[:1] == sealed_left
