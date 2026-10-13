from dataclasses import replace
import pytest
from src.trackocd_cvpr.self_excluded_mean_control import SelfExcludedMeanMemberControl
from src.trackocd_cvpr.effective_excluded_mean_audit import audit_chosen_excluded_mean
from src.trackocd_cvpr.versioned_causal_outcome_audit import VersionedCausalOutcomeAudit
from tests.trackocd_cvpr.test_versioned_causal_outcome_audit import anon_only
from tests.trackocd_cvpr.test_versioned_reliability_memory import obs
import numpy as np


def fixture(known_label=('Known', 4), aliases=('past', 'current', 'query')):
    known = {0: np.eye(1, 768, 7, dtype=np.float32)[0]}
    labels = {(10, aliases[0]): ('declared_pseudoNovel', 9), (20, aliases[1]): known_label,
              (20, aliases[2]): ('declared_pseudoNovel', 9)}
    control = SelfExcludedMeanMemberControl(known, anon_only, exclusion_mode='exclude_current_video')
    control.ingest(0, [obs(aliases[0])]); control.start_next_video()
    control.ingest(0, [obs(aliases[1], 1)])
    prior, _ = control.state_snapshot(); before_history = control.memory_versions()
    result = control.ingest(1, [obs(aliases[2])]); receipt, = control.query_view_receipts()
    assert result.events[0].action == 'EXISTING'
    return control, prior, result, receipt, labels, before_history


def test_true_selected_past_support_can_be_clean_while_whole_original_history_stays_dirty():
    control, prior, result, receipt, labels, history = fixture()
    truth = audit_chosen_excluded_mean(prior, ('query',), receipt, result.events[0].token, labels, {0: 10, 1: 20}, frame_index=1)
    assert truth.status == 'CERTIFIED_ONE_DECLARED_PSEUDO_EFFECTIVE_SUPPORT_LOWER_BOUND'
    assert truth.selected_members == frozenset([(10, 'past')])
    assert truth.independent_reuse_bounds(9, (20, 'query'))['selected_cross_video_member_reuse_lower_bound']
    assert not truth.original_history_repaired_or_full_semantic_truth_or_statistical_independence_accepted
    assert control.memory_versions()[:len(history)] == history
    # The original chosen version remains dirty: active truth is not relabeling
    # its ancestor history or repairing an earlier prediction.
    from src.trackocd_cvpr.version_lineage_audit import audit_version_lineage
    targets = {(v, r): ('policy_pseudo', 0) if kind == 'declared_pseudoNovel' else (kind, category)
               for (v, r), (kind, category) in labels.items()}
    bank, _ = control.state_snapshot()
    original = audit_version_lineage(control.memory_versions(), bank.commit_history(), targets, {0: 10, 1: 20})
    assert original['strict_latest_version_status_counts']['VERIFIED_BAD_ANONYMOUS_KNOWN_HISTORY'] == 1


def test_unknown_past_support_never_becomes_certified_by_excluding_current_Known():
    control, prior, result, receipt, labels, _ = fixture()
    labels[(10, 'past')] = ('no_retained_semantic_truth', None)
    truth = audit_chosen_excluded_mean(prior, ('query',), receipt, result.events[0].token, labels, {0: 10, 1: 20}, frame_index=1)
    assert truth.status == 'UNSCORABLE_EFFECTIVE_UNKNOWN_SUPPORT'
    assert not truth.certified_for_declared_pseudo(9)


def test_receipt_mismatch_precedes_any_semantic_lookup_or_partial_purity_acceptance():
    _, prior, result, receipt, _, _ = fixture()
    class ForbiddenLabels(dict):
        def __contains__(self, key): raise AssertionError('GT lookup before numeric receipt validated')
    with pytest.raises(ValueError, match='receipt differs'):
        audit_chosen_excluded_mean(prior, ('query',), replace(receipt, numeric_states_sha256='0' * 64),
            result.events[0].token, ForbiddenLabels(), {0: 10, 1: 20}, frame_index=1)


@pytest.mark.parametrize('fault', ['future_epoch', 'missing_member', 'missing_token', 'current_frame'])
def test_whole_membership_epoch_timing_and_numeric_chosen_binding_required(fault):
    _, prior, result, receipt, labels, _ = fixture(); epochs = {0: 10, 1: 20}; token = result.events[0].token; frame = 1
    if fault == 'future_epoch': epochs[2] = 30
    elif fault == 'missing_member': labels.pop((10, 'past'))
    elif fault == 'missing_token': token = 'S:999'
    else: frame = 0
    with pytest.raises(ValueError): audit_chosen_excluded_mean(prior, ('query',), receipt, token, labels, epochs, frame_index=frame)


def test_actual_ID_bijection_only_changes_private_correspondence_not_purity_or_numeric_prototype():
    truths = []
    for aliases in (('past', 'current', 'query'), ('9', 'alpha', 'z')):
        _, prior, result, receipt, labels, _ = fixture(aliases=aliases)
        truths.append(audit_chosen_excluded_mean(prior, (aliases[2],), receipt, result.events[0].token,
            labels, {0: 10, 1: 20}, frame_index=1))
    assert truths[0].status == truths[1].status and truths[0].numeric_prototype_sha256 == truths[1].numeric_prototype_sha256
    assert truths[0].selected_groups == truths[1].selected_groups == 1


def test_Known_retained_effective_member_is_verified_bad_not_unknown_or_clean():
    _, prior, result, receipt, labels, _ = fixture()
    labels[(10, 'past')] = ('Known', 4)
    truth = audit_chosen_excluded_mean(prior, ('query',), receipt, result.events[0].token, labels, {0: 10, 1: 20}, frame_index=1)
    assert truth.status == 'VERIFIED_BAD_EFFECTIVE_ANONYMOUS_KNOWN_SUPPORT'
    assert not truth.certified_for_declared_pseudo(9)


def test_retained_mixed_declared_classes_are_not_certified_from_one_good_member():
    known = {0: np.eye(1, 768, 7, dtype=np.float32)[0]}
    c = SelfExcludedMeanMemberControl(known, anon_only, exclusion_mode='exclude_current_video')
    c.ingest(0, [obs('a')]); c.start_next_video(); c.ingest(0, [obs('b', 1)]); c.start_next_video()
    prior, _ = c.state_snapshot(); result = c.ingest(0, [obs('q')]); receipt, = c.query_view_receipts()
    labels = {(10, 'a'): ('declared_pseudoNovel', 9), (20, 'b'): ('declared_pseudoNovel', 10)}
    truth = audit_chosen_excluded_mean(prior, ('q',), receipt, result.events[0].token, labels, {0: 10, 1: 20, 2: 30}, frame_index=0)
    assert truth.status == 'VERIFIED_BAD_EFFECTIVE_MIXED_DECLARED_PSEUDO_SUPPORT'
    assert truth.selected_groups == 2 and not truth.certified_for_declared_pseudo(9)


def test_ancestor_other_video_does_not_become_current_cross_video_numeric_support():
    from src.trackocd_cvpr.C_development_rollout import raw_fixed_scores
    known = {0: np.eye(1, 768, 7, dtype=np.float32)[0]}
    c = SelfExcludedMeanMemberControl(known, lambda f, x: raw_fixed_scores(f, x, tau_known=.65, tau_existing=.55),
        exclusion_mode='exclude_query_alias_scope')
    labels = {(10, 'a'): ('declared_pseudoNovel', 9), (20, 'b'): ('declared_pseudoNovel', 9),
              (20, 'c'): ('declared_pseudoNovel', 9), (20, 'q'): ('declared_pseudoNovel', 9), (20, 'known'): ('Known', 4)}
    audit = VersionedCausalOutcomeAudit(labels, {0: 4})
    def run(video, frame, rows):
        r = c.ingest(frame, rows); bank, _ = c.state_snapshot()
        audit.audit_frame(video, r, c.chosen_version_bindings(), c.memory_versions(), bank.commit_history())
        return r
    audit.start_video(0, 10); run(10, 0, [obs('a')])
    c.start_next_video(); audit.start_video(1, 20)
    run(20, 0, [replace(obs('b'), quality=.1), obs('c', 1)])
    moved = run(20, 1, [replace(obs('b', 1), quality=.9)])
    assert moved.events[0].action == 'EXISTING' and moved.events[0].token == 'S:1'
    prior, _ = c.state_snapshot(); r = run(20, 2, [obs('q', 1), obs('known', 7)])
    ordinal = next(i for i, routes in enumerate(r.routing_groups) if routes == ('q',))
    truth = audit_chosen_excluded_mean(prior, ('q',), c.query_view_receipts()[ordinal], r.events[ordinal].token,
        labels, {0: 10, 1: 20}, frame_index=2)
    assert truth.selected_members == frozenset([(20, 'b'), (20, 'c')])
    assert truth.independent_reuse_bounds(9, (20, 'q'))['selected_other_physical_member_reuse_lower_bound']
    assert not truth.independent_reuse_bounds(9, (20, 'q'))['selected_cross_video_member_reuse_lower_bound']
    historical = audit.finish()['chosen_preframe_version_prefix_snapshots']['all_available']
    assert historical['certified_cross_video_reuse_micro_lower_bound'] > 0
