import json
import pytest
from src.trackocd_cvpr.fit_loss_supervision import project_fit_loss_targets, external_targets_for_physical_tracks


def fixture(tmp_path):
    assignment = tmp_path / 'assignment.json'
    assignment.write_text(json.dumps({'assignment': {'classes': {'fit_known': [3, 7],
        'development_pseudo': {'must_not_decode': True}, 'independent_final_pseudo': {'must_not_decode': True}},
        'video_ids': {'fit_known': [10, 11], 'development_pseudo': {'must_not_decode': True}}}}))
    known = tmp_path / 'known.json'
    known.write_text(json.dumps({'tracks': [
        {'video_id': 99, 'category_id': {'protected': True}, 'track_id': {'protected': True}},
        {'video_id': 10, 'category_id': 3, 'track_id': 1, 'annotation_count': 3},
        {'video_id': 11, 'category_id': 7, 'track_id': 2, 'annotation_count': 4},
        {'video_id': 10, 'category_id': 19, 'track_id': {'not_needed': True}, 'annotation_count': {'not_needed': True}}]}))
    return assignment, known


def test_fit_only_byte_projection_external_loss_targets_and_no_unknown_BG(tmp_path):
    assignment, known = fixture(tmp_path)
    bindings, targets, proof = project_fit_loss_targets(assignment, known, expected_classes=2, expected_videos=2, expected_tracks=2)
    assert bindings == {3: 0, 7: 1} and targets == {(10, 1): 0, (11, 2): 1}
    assert external_targets_for_physical_tracks(10, [1, 999], targets) == (0, None)
    assert proof['eligible_supervised_observations'] == 7
    assert proof['fit_other_retained_Known_tracks_without_loss_target'] == 1
    assert not proof['other_role_semantic_values_decoded']
    assert not proof['unlabeled_or_other_Known_assigned_background_or_Novel']


def test_fixed_scope_and_supervision_counts_are_not_silently_relaxed(tmp_path):
    assignment, known = fixture(tmp_path)
    with pytest.raises(ValueError, match='counts'):
        project_fit_loss_targets(assignment, known)
    with pytest.raises(ValueError, match='denominator'):
        project_fit_loss_targets(assignment, known, expected_classes=2, expected_videos=2, expected_tracks=3)
