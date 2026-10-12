from copy import deepcopy
import pytest
from src.trackocd_cvpr.global_protocol_coverage import audit_global_coverage
from src.trackocd_cvpr.joint_known_protocol import verify_global_assignment
from src.trackocd_cvpr.known_census_protocol import ROLES
from src.trackocd_cvpr.global_protocol_coverage import verified_global_wrapper_assignment


def data():
    presence = {4 * c + i: {c} for c in range(14) for i in range(4)}
    groups = (list(range(2)), [2, 3], [4, 5], list(range(6, 14)))
    assignment = {'classes': dict(zip(ROLES, groups)), 'video_ids': {
        r: [4 * c + i for c in cs for i in range(3 if r == ROLES[0] else 2)]
        for r, cs in zip(ROLES, groups)}}
    index = {'tracks': [{'category_id': next(iter(cs)), 'video_id': v,
                         'annotation_count': 1, 'track_id': v, 'distinct_annotated_images': 1}
                        for v, cs in presence.items()]}
    return presence, index, assignment


def test_all_unique_legal_videos_added_counts_cross_video_queries_no_freeze():
    p, index, assignment = data()
    saved = deepcopy(assignment)
    public, private = audit_global_coverage(p, index, assignment)
    assert public['base_assigned_videos'] == 30 and public['expanded_assigned_videos'] == 56
    assert public['added_unique_role_videos'] == dict(fit_known=2, policy_train_pseudo=4,
        development_pseudo=4, independent_final_pseudo=16)
    assert public['expanded_support']['independent_final_pseudo']['possible_cross_video_query_tracks']['sum'] == 32
    assert public['expanded_support']['independent_final_pseudo']['single_annotation_tracks'] == 32
    assert verify_global_assignment(p, private['assignment'])
    assert not public['new_protocol_frozen'] and not public['class_assignment_changed']
    assert assignment == saved


def test_conflicting_video_not_silently_assigned_and_nonpool_not_background():
    p, index, assignment = data()
    p[1000] = {2, 4}; p[1001] = {999}
    public, private = audit_global_coverage(p, index, assignment)
    assert public['unassigned_no_legal_positive_own_role_videos'] == 2
    assert public['unassigned_ambiguous_positive_role_videos'] == 0
    assert all(1000 not in vs and 1001 not in vs for vs in private['assignment']['video_ids'].values())
    assert not public['complete_visual_category_absence_proven']


def test_extension_idempotent_all_remaining_candidates_accounted():
    p, index, assignment = data()
    first, private = audit_global_coverage(p, index, assignment)
    second, _ = audit_global_coverage(p, index, private['assignment'])
    assert sum(second['added_unique_role_videos'].values()) == 0
    assert first['expanded_support'] == second['expanded_support']
    assert (second['expanded_assigned_videos'] + second['unassigned_ambiguous_positive_role_videos']
            + second['unassigned_no_legal_positive_own_role_videos']) == len(p)


def test_actual_writer_three_field_wrapper_and_flag_unknown_schema_refusals():
    _, _, assignment = data()
    value = {'not_model_facing': True, 'not_a_frozen_protocol': True, 'assignment': assignment}
    assert verified_global_wrapper_assignment(value) is assignment
    for bad in ({'not_model_facing': True, 'assignment': assignment},
                {**value, 'not_a_frozen_protocol': False}, {**value, 'GT': []}):
        with pytest.raises(ValueError):
            verified_global_wrapper_assignment(bad)
