import numpy as np
from src.trackocd_cvpr.causal_category_audit import CausalCategoryAudit
from src.trackocd_cvpr.atomic_member_control import AtomicMemberControl
from tests.trackocd_cvpr.test_atomic_member_control import obs


def raw_scores(features, context):
    return np.r_[features[:, 0], .5, -2.].astype(np.float32)


def controller(): return AtomicMemberControl({0: obs('prior', 7).visual}, raw_scores)


def test_full_physical_unknown_and_short_prefixes_not_dropped_and_crossvideo_not_self():
    labels = {(1, 'a'): ('declared_pseudoNovel', 9), (2, 'b'): ('declared_pseudoNovel', 9),
              (3, 'k'): ('Known', 7), (4, 'u'): ('no_retained_semantic_truth', None)}
    c = controller(); audit = CausalCategoryAudit(labels, {0: 7})
    for i, (video, route, axis) in enumerate(((1, 'a', 0), (2, 'b', 0), (3, 'k', 7), (4, 'u', 2))):
        if i: c.start_next_video()
        audit.audit_frame(video, c.ingest(0, [obs(route, axis)]))
    r = audit.finish(); full = r['prefix_snapshots']['all_available']
    assert full['physical_tracks'] == 4 and full['Known_queries'] == 1 and full['declared_pseudo_queries'] == 2
    assert full['Known_micro_accuracy'] == 1 and full['certified_cross_video_reuse_micro_lower_bound'] == .5
    assert full['all_physical_commit_track_coverage'] == 1
    assert r['prefix_snapshots']['16'] == full  # All singleton tracks stay in denominator.
    assert r['strict_route_reassignment_history']['semantic_unscorable_history'] == 1
    assert not r['unknown_declared_background_or_Novel']


def test_actual_same_frame_peer_pollution_cannot_rewrite_preframe_reuse_or_feedback_into_bank():
    labels = {(1, 'a'): ('declared_pseudoNovel', 9), (2, 'b'): ('declared_pseudoNovel', 9), (2, 'k'): ('Known', 7)}
    c = controller(); audit = CausalCategoryAudit(labels, {0: 7})
    audit.audit_frame(1, c.ingest(0, [obs('a')]))
    c.start_next_video(); frame = c.ingest(0, [obs('b'), obs('k')])
    before = c.state_snapshot()[0].commit_history()
    audit.audit_frame(2, frame)
    assert c.state_snapshot()[0].commit_history() == before
    r = audit.finish()
    assert r['prefix_snapshots']['all_available']['certified_cross_video_reuse_micro_lower_bound'] == .5
    assert r['strict_route_reassignment_history']['verified_non_pseudo_or_mixed_history'] == 1
    assert r['first_Known_anonymous_write_canonical_group'] == 1


def test_semantic_unknown_existing_state_not_declared_background_or_certified_bad():
    labels = {(1, 'u'): ('no_retained_semantic_truth', None), (2, 'p'): ('declared_pseudoNovel', 9), (3, 'k'): ('Known', 7)}
    c = controller(); audit = CausalCategoryAudit(labels, {0: 7})
    for i, (video, route, axis) in enumerate(((1, 'u', 0), (2, 'p', 0), (3, 'k', 7))):
        if i: c.start_next_video()
        audit.audit_frame(video, c.ingest(0, [obs(route, axis)]))
    full = audit.finish()['prefix_snapshots']['all_available']
    assert full['certified_cross_video_reuse_micro_lower_bound'] == 0
    assert full['unscorable_reuse_queries_retained'] == 1 and full['physical_tracks'] == 3


def test_actual_route_move_keeps_parent_pollution_and_reassignment_cycles_terminate():
    from tests.trackocd_cvpr.test_atomic_member_control import new_policy, reuse_policy
    calls = []
    def policy(features, context):
        calls.append(1)
        if len(calls) == 4:
            values = np.full(len(features) + 2, -2., dtype=np.float32)
            values[int(np.argmin(features[:, 0]))] = 2.
            return values
        return (new_policy if len(calls) in (1, 3) else reuse_policy)(features, context)
    c = AtomicMemberControl({}, policy)
    audit = CausalCategoryAudit({(1, 'k'): ('Known', 7), (2, 'p'): ('declared_pseudoNovel', 9)}, {0: 7})
    audit.audit_frame(1, c.ingest(0, [obs('k')]))
    c.start_next_video(); audit.audit_frame(2, c.ingest(0, [obs('p')]))
    audit.audit_frame(2, c.ingest(1, [obs('p', 1)]))  # p moves S:0 -> S:1, old contamination survives.
    assert not audit.token_truth('S:1', include_route_ancestors=False).has_Known
    assert audit.token_truth('S:1').has_Known
    # An actual later token reassignment can create a provenance cycle. Audit
    # closure must terminate and retain truth, not remove edges to wash history.
    last = c.ingest(2, [obs('p', 1)])
    assert last.events[0].token == 'S:0'
    audit.audit_frame(2, last)
    assert audit.token_truth('S:0').has_Known and audit.token_truth('S:1').has_Known
    assert audit.finish()['strict_route_reassignment_history']['verified_non_pseudo_or_mixed_history'] == 2


def test_different_external_truth_and_ID_rename_cannot_change_actual_numeric_policy_events():
    left, right = controller(), controller()
    a = CausalCategoryAudit({(1, 'a'): ('Known', 7), (2, 'b'): ('declared_pseudoNovel', 9)}, {0: 7})
    b = CausalCategoryAudit({(1, 'renamed'): ('declared_pseudoNovel', 9), (2, 'other'): ('Known', 7)}, {0: 7})
    first_l = left.ingest(0, [obs('a')]); first_r = right.ingest(0, [obs('renamed')])
    assert first_l.events == first_r.events
    a.audit_frame(1, first_l); b.audit_frame(1, first_r)
    left.start_next_video(); right.start_next_video()
    second_l = left.ingest(0, [obs('b')]); second_r = right.ingest(0, [obs('other')])
    assert second_l.events == second_r.events
    a.audit_frame(2, second_l); b.audit_frame(2, second_r)
    assert left.state_snapshot()[0].state_evidence() == right.state_snapshot()[0].state_evidence()
