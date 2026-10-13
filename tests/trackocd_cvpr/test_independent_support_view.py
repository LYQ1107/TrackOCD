from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from src.trackocd_cvpr.causal_observation_router import CurrentObservationRouter
from src.trackocd_cvpr.prefix_member_bank import PrefixMemberBank
from src.trackocd_cvpr.independent_support_view import past_support_excluding
from src.trackocd_cvpr.reliability_support_bridge import committed_reliability_inputs
from tests.trackocd_cvpr.test_versioned_reliability_memory import obs


def fixture(aliases=('query', 'duplicate', 'other')):
    bank = PrefixMemberBank(); router = CurrentObservationRouter()
    batch = router.ingest(0, [obs(aliases[0]), obs(aliases[1])])
    bank.apply_frame(batch, ['S:0'])
    # Exact payload in another VIDEO is not the same alias scope.
    bank.start_next_video(); router = CurrentObservationRouter()
    batch = router.ingest(0, [obs(aliases[2])]); bank.apply_frame(batch, ['S:0'])
    return bank


def test_current_video_exclusion_keeps_past_scope_not_cumulative_votes_or_empty_purity():
    bank = fixture(); original = bank.raw_support_snapshot(); commits = bank.commit_history()
    view = past_support_excluding(bank, ('other',), mode='exclude_current_video')
    assert (view.original_groups, view.retained_groups, view.original_routed_members, view.retained_routed_members) == (2, 1, 3, 2)
    assert len(view.supports[0].payloads) == 1
    assert bank.raw_support_snapshot() == original and bank.commit_history() == commits
    assert not view.independent_semantic_purity_or_full_self_excluded_evaluation_accepted


def test_query_alias_exclusion_removes_whole_old_alias_scope_not_one_id():
    bank = PrefixMemberBank(); router = CurrentObservationRouter()
    batch = router.ingest(0, [obs('query'), obs('unlisted-exact-alias')]); bank.apply_frame(batch, ['S:0'])
    view = past_support_excluding(bank, ('query',), mode='exclude_query_alias_scope')
    assert view.original_groups == 1 and view.retained_groups == view.retained_routed_members == 0
    assert view.supports[0].token == 'S:0' and not view.supports[0].payloads  # explicit tombstone
    assert len(bank.commit_history()) == 2  # no real memory repair or deletion


def test_ID_bijection_changes_only_selection_handles_not_numeric_support():
    left = past_support_excluding(fixture(), ('other',), mode='exclude_current_video')
    right = past_support_excluding(fixture(('z', 'alpha', '9')), ('9',), mode='exclude_current_video')
    assert left == right
    for a, b in zip(committed_reliability_inputs(left.supports, 'S:0'), committed_reliability_inputs(right.supports, 'S:0')):
        assert np.array_equal(a.numpy(), b.numpy())


def test_new_query_missing_past_binding_keeps_scope_but_never_claims_independence():
    bank = fixture(); view = past_support_excluding(bank, ('new-query',), mode='exclude_query_alias_scope')
    assert view.original_groups == view.retained_groups == 2
    assert not view.independent_semantic_purity_or_full_self_excluded_evaluation_accepted


def test_unknown_mode_or_duplicate_query_is_refused_without_bank_changes():
    bank = fixture(); before = deepcopy(bank).raw_support_snapshot()
    for queries, mode in ((('other',), 'GT-clean'), (('other', 'other'), 'exclude_current_video'), ((), 'exclude_current_video')):
        with pytest.raises(ValueError): past_support_excluding(bank, queries, mode=mode)
    assert bank.raw_support_snapshot() == before


def test_fabricated_raw_payload_not_bound_to_real_last_commit_is_refused():
    bank = fixture(); key = next(iter(bank._members))
    bank._members[key] = replace(bank._members[key], payload=b'not-the-committed-visible-prefix')
    with pytest.raises(ValueError, match='last committed'):
        past_support_excluding(bank, ('other',), mode='exclude_current_video')
