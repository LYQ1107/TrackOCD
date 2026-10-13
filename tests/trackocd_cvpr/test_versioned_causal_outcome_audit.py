from dataclasses import replace
import numpy as np
import pytest
from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
from src.trackocd_cvpr.reliability_member_control import ReliabilityMemberControl
from src.trackocd_cvpr.frame_atomic_state_ema import StateEMAMemberControl
from src.trackocd_cvpr.versioned_causal_outcome_audit import VersionedCausalOutcomeAudit
from tests.trackocd_cvpr.test_versioned_reliability_memory import obs


def anon_only(f, c):
    s = np.full(len(f) + 2, -2., np.float32)
    existing = np.flatnonzero(f[:, 1] == 0)
    s[int(existing[0]) if len(existing) else -2] = 1
    return s


def fixture(control_class=ReliabilityMemberControl):
    vector = np.zeros(768, np.float32); vector[7] = 1
    control = (control_class({0: vector}, anon_only, MemberReliabilityMLP())
               if control_class is ReliabilityMemberControl else control_class({0: vector}, anon_only))
    return control


def audit_actual(audit, control, video, frame, rows):
    result = control.ingest(frame, rows)
    bank, _ = control.state_snapshot()
    audit.audit_frame(video, result, control.chosen_version_bindings(), control.memory_versions(), bank.commit_history())
    return result


def test_actual_M_chosen_version_prior_certificate_not_current_peer_or_future_truth():
    labels = {(10, 'a'): ('declared_pseudoNovel', 20), (20, 'b'): ('declared_pseudoNovel', 20),
              (30, 'known'): ('Known', 10)}
    control = fixture(); audit = VersionedCausalOutcomeAudit(labels, {0: 10})
    audit.start_video(0, 10); audit_actual(audit, control, 10, 0, [obs('a')])
    control.start_next_video(); audit.start_video(1, 20)
    audit_actual(audit, control, 20, 0, [obs('b')])
    control.start_next_video(); audit.start_video(2, 30)
    audit_actual(audit, control, 30, 0, [obs('known')])
    result = audit.finish(); end = result['chosen_preframe_version_prefix_snapshots']['all_available']
    assert end['declared_pseudo_queries'] == 2 and end['certified_cross_video_reuse_micro_lower_bound'] == .5
    # Late Known pollution makes final history bad, but never rewrites earlier
    # sealed reuse of the genuinely pure prior version.
    assert result['strict_latest_version_status_counts'] == {'VERIFIED_BAD_ANONYMOUS_KNOWN_HISTORY': 1}
    assert result['strict_all_version_status_counts']['CERTIFIED_ONE_DECLARED_PSEUDO_CLASS_LOWER_BOUND'] == 2
    assert not result['current_active_purity_official_CT_full_semantic_truth_Pred_or_self_exclusion_claim']


def test_actual_EMA_same_external_version_interface_unknown_retained_not_background():
    labels = {(1, 'unknown'): ('no_retained_semantic_truth', None),
              (2, 'pseudo'): ('declared_pseudoNovel', 20), (3, 'known'): ('Known', 10)}
    control = fixture(StateEMAMemberControl); audit = VersionedCausalOutcomeAudit(labels, {0: 10})
    for epoch, (video, route) in enumerate(((1, 'unknown'), (2, 'pseudo'), (3, 'known'))):
        if epoch: control.start_next_video()
        audit.start_video(epoch, video); audit_actual(audit, control, video, 0, [obs(route)])
    end = audit.finish()['chosen_preframe_version_prefix_snapshots']['all_available']
    assert end['physical_tracks'] == 3 and end['all_physical_commit_track_coverage'] == 1
    assert end['certified_cross_video_reuse_micro_lower_bound'] == 0
    assert end['unscorable_reuse_queries_retained'] == 2


def test_postframe_or_missing_chosen_version_never_uses_current_gt_peer_to_certify():
    labels = {(1, 'a'): ('declared_pseudoNovel', 20), (2, 'b'): ('declared_pseudoNovel', 20),
              (3, 'known'): ('Known', 10)}
    control = fixture(); audit = VersionedCausalOutcomeAudit(labels, {0: 10})
    audit.start_video(0, 1); audit_actual(audit, control, 1, 0, [obs('a')])
    control.start_next_video(); audit.start_video(1, 2)
    result = control.ingest(0, [obs('b')]); bank, _ = control.state_snapshot()
    future_ref = (control.memory_versions()[-1].token, control.memory_versions()[-1].revision)
    with pytest.raises(ValueError, match='pre-frame'):
        audit.audit_frame(2, result, [future_ref], control.memory_versions(), bank.commit_history())
    with pytest.raises(ValueError, match='Failed whole'): audit.finish()


def test_ID_bijection_changes_external_bit_positions_not_numerical_or_certified_outcomes():
    results = []
    for aliases in (('1', '2', '3'), ('z', 'alpha', '9')):
        labels = {(1, aliases[0]): ('declared_pseudoNovel', 20),
                  (2, aliases[1]): ('declared_pseudoNovel', 20), (3, aliases[2]): ('Known', 10)}
        control = fixture(); audit = VersionedCausalOutcomeAudit(labels, {0: 10})
        for epoch, (video, route) in enumerate(zip((1, 2, 3), aliases)):
            if epoch: control.start_next_video()
            audit.start_video(epoch, video); audit_actual(audit, control, video, 0, [obs(route)])
        results.append(audit.finish())
    assert results[0] == results[1]


def test_version_bound_refuses_whole_audit_not_dirty_ancestor_eviction():
    labels = {(1, 'a'): ('declared_pseudoNovel', 20), (2, 'b'): ('declared_pseudoNovel', 20), (3, 'known'): ('Known', 10)}
    control = fixture(); audit = VersionedCausalOutcomeAudit(labels, {0: 10}, max_versions=1)
    audit.start_video(0, 1); audit_actual(audit, control, 1, 0, [obs('a')])
    control.start_next_video(); audit.start_video(1, 2)
    with pytest.raises(ValueError): audit_actual(audit, control, 2, 0, [obs('b')])
    with pytest.raises(ValueError, match='Failed whole'): audit.finish()
