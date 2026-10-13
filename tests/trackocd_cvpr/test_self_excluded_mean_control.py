from copy import deepcopy
import numpy as np
import pytest
from src.trackocd_cvpr.causal_observation_router import CurrentObservationRouter
from src.trackocd_cvpr.frame_atomic_state_ema import FrameAtomicStateEMA, StateEMAMemberControl
from src.trackocd_cvpr.prefix_member_bank import PrefixMemberBank
from src.trackocd_cvpr.self_excluded_mean_control import preframe_excluded_mean, SelfExcludedMeanMemberControl
from tests.trackocd_cvpr.test_versioned_reliability_memory import obs, reuse


def test_query_alias_cannot_leave_duplicate_self_vote_or_revive_empty_candidate():
    control = SelfExcludedMeanMemberControl({}, reuse, exclusion_mode='exclude_query_alias_scope')
    control.ingest(0, [obs('query'), obs('alias')])
    saved = control.memory_versions()
    result = control.ingest(1, [obs('query')])
    assert result.events[0].action == 'NEW'  # original Mean would reuse S:0
    receipt, = control.query_view_receipts()
    assert (receipt.original_groups, receipt.retained_groups) == (1, 0)
    assert (receipt.original_routed_members, receipt.retained_routed_members) == (2, 0)
    assert not receipt.independent_semantic_purity_or_actual_development_accepted
    assert control.memory_versions()[:len(saved)] == saved
    bank, _ = control.state_snapshot()
    assert len(bank.commit_history()) == 3  # dirty history never deleted
    assert len(bank.raw_support_snapshot()) == 2


def test_current_video_exclusion_no_peer_write_but_cross_video_reuse_is_available():
    control = SelfExcludedMeanMemberControl({}, reuse, exclusion_mode='exclude_current_video')
    first = control.ingest(0, [obs('a'), obs('b', 1)])
    assert all(e.action == 'NEW' for e in first.events)
    assert all(r.retained_groups == 0 for r in control.query_view_receipts())
    result = control.ingest(1, [obs('a'), obs('new-peer', 2)])
    assert all(e.action == 'NEW' for e in result.events)
    control.start_next_video()
    # Two axis0 tokens would be indistinguishable; inspect numeric views rather
    # than claim an Existing selection from arbitrary token tie breaking.
    control.ingest(0, [obs('next-video', 3)])
    receipt, = control.query_view_receipts()
    assert receipt.retained_groups == receipt.original_groups == 3  # latest-prefix replacement, not 4 votes
    assert receipt.original_version_reference_is_conservative_history_only


def test_cross_video_reuse_binds_old_dirty_version_and_separate_numeric_receipt():
    control = SelfExcludedMeanMemberControl({}, reuse, exclusion_mode='exclude_current_video')
    control.ingest(0, [obs('a')]); old = control.memory_versions()
    control.start_next_video(); result = control.ingest(0, [obs('b')])
    assert result.events[0].action == 'EXISTING'
    assert control.chosen_version_bindings() == (('S:0', 0),)
    assert control.query_view_receipts()[0].retained_groups == 1
    assert control.memory_versions()[:len(old)] == old


def test_no_exclusion_recomputes_mean_exactly_and_does_not_use_stale_latent():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('a'), obs('alias')]), ['S:0'])
    bank.start_next_video(); router = CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('b', 1)]), ['S:0'])
    before = deepcopy(bank)
    expected = FrameAtomicStateEMA(momentum=0.).refresh(bank)['S:0']
    states, receipt = preframe_excluded_mean(bank, ('new',), mode='exclude_query_alias_scope', frame_index=1)
    assert states['S:0'] == expected
    assert states['S:0'].routed_members == 3 and states['S:0'].support_groups == 2
    assert bank.raw_support_snapshot() == before.raw_support_snapshot()
    assert bank.commit_history() == before.commit_history()
    filtered, _ = preframe_excluded_mean(bank, ('b',), mode='exclude_current_video', frame_index=1)
    np.testing.assert_array_equal(filtered['S:0'].prototype(), np.eye(1, 768)[0])
    assert filtered['S:0'].routed_members == 2 and receipt.retained_groups == 2


@pytest.mark.parametrize('mode', ['exclude_query_alias_scope', 'exclude_current_video'])
def test_actual_controller_ID_bijection_reverse_delivery_and_empty_clock(mode):
    left = SelfExcludedMeanMemberControl({}, reuse, exclusion_mode=mode)
    right = SelfExcludedMeanMemberControl({}, reuse, exclusion_mode=mode)
    for frame, a, b in ((0, [obs('q'), obs('alias')], [obs('z'), obs('9')]),
                        (1, [obs('q', 1), obs('peer', 2)], [obs('foo', 2), obs('9', 1)]),
                        (2, [], [])):
        assert left.ingest(frame, a).events == right.ingest(frame, b).events
        assert left.query_view_receipts() == right.query_view_receipts()
        assert left.memory_versions() == right.memory_versions()
    assert left.query_view_receipts() == ()


def test_frame_failure_rolls_back_router_births_original_history_and_view_receipts():
    control = SelfExcludedMeanMemberControl({}, reuse, exclusion_mode='exclude_current_video', max_versions=1)
    with pytest.raises(ValueError, match='version bound'):
        control.ingest(0, [obs('a'), obs('b', 1)])
    bank, next_id = control.state_snapshot()
    assert bank.commit_history() == () and next_id == 0
    assert control.memory_versions() == control.query_view_receipts() == ()
    control.ingest(0, [obs('a')])


def test_preframe_timing_and_EMA_injection_refused():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(4, [obs('a')]), ['S:0'])
    for frame in (-1, 0, 4, 4.):
        with pytest.raises(ValueError, match='pre-frame'):
            preframe_excluded_mean(bank, ('a',), mode='exclude_query_alias_scope', frame_index=frame)
    with pytest.raises(ValueError, match='no EMA'):
        SelfExcludedMeanMemberControl({}, reuse, exclusion_mode='exclude_current_video', momentum=.9)


def test_new_unknown_query_matches_original_mean_decisions_not_claimed_scope_independence():
    excluded = SelfExcludedMeanMemberControl({}, reuse, exclusion_mode='exclude_query_alias_scope')
    original = StateEMAMemberControl({}, reuse, momentum=0.)
    for frame, rows in ((0, [obs('a')]), (1, [obs('new-b', 1)])):
        assert excluded.ingest(frame, rows).events == original.ingest(frame, rows).events
        assert excluded.memory_versions() == original.memory_versions()


def test_quality_statistics_use_exact_original_group_aggregation_not_alias_weighting():
    from dataclasses import replace
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    batch = router.ingest(0, [replace(obs('a'), quality=.123456),
                             replace(obs('b', 1), quality=.765432),
                             replace(obs('b-alias', 1), quality=.765432)])
    bank.apply_frame(batch, ['S:0'] * len(batch.model_views))
    states, _ = preframe_excluded_mean(bank, ('unbound',), mode='exclude_query_alias_scope', frame_index=1)
    assert states['S:0'] == FrameAtomicStateEMA(momentum=0.).refresh(bank)['S:0']


def test_future_suffix_change_cannot_rewrite_sealed_earlier_events_or_query_views():
    def run(suffix):
        c = SelfExcludedMeanMemberControl({}, reuse, exclusion_mode='exclude_query_alias_scope')
        earlier = []
        for frame, rows in ((0, [obs('a')]), (1, [obs('a', 1)])):
            result = c.ingest(frame, rows)
            earlier.append((result.events, c.query_view_receipts(), c.memory_versions()))
        c.ingest(2, [obs('a', suffix)])
        return earlier
    assert run(2) == run(7)


def test_nonfinite_numeric_scorer_refuses_entire_frame_without_partial_peer_commit():
    calls = []
    def bad(f, c):
        calls.append(len(f))
        return reuse(f, c) if len(calls) == 1 else np.full(len(f) + 2, np.nan)
    control = SelfExcludedMeanMemberControl({}, bad, exclusion_mode='exclude_current_video')
    with pytest.raises(ValueError, match='score'):
        control.ingest(0, [obs('a'), obs('b', 1)])
    assert calls == [0, 0]
    assert control.state_snapshot()[0].commit_history() == ()
    assert control.state_snapshot()[1] == 0
    assert control.query_view_receipts() == control.memory_versions() == ()


def test_source_contract_pins_private_provenance_and_does_not_accept_experiments():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / 'configs/trackocd_cvpr/source_self_excluded_mean_control_v1.json').read_text())
    assert hashlib.sha256((root / cfg['pinned_bank_dependency']).read_bytes()).hexdigest() == cfg['pinned_bank_sha256']
    assert not cfg['EMA_or_trained_M_self_exclusion']
    assert not cfg['semantic_independence_purity_or_full_physical_semantic_truth_accepted']
    assert not cfg['automatic_private_data_read_training_or_experiment_launch']
