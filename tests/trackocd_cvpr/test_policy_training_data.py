import json
from dataclasses import fields

import numpy as np
import pytest

from src.trackocd_cvpr.policy_training_data import (
    project_policy_loss_targets, reconcile_policy_targets, policy_current_frames,
)
from src.trackocd_cvpr.causal_observation_router import CurrentObservation, CurrentObservationRouter, visible_bytes


def fixture(tmp_path):
    assignment = tmp_path / 'assignment.json'; known = tmp_path / 'known.json'
    assignment.write_text(json.dumps({'assignment': {'classes': {
        'fit_known': [3], 'policy_train_pseudo': [7],
        'development_pseudo': {'cannot_decode': True},
        'independent_final_pseudo': {'cannot_decode': True}}, 'video_ids': {
        'policy_train_pseudo': [10], 'fit_known': {'cannot_decode': True},
        'development_pseudo': {'cannot_decode': True},
        'independent_final_pseudo': {'cannot_decode': True}}}}))
    known.write_text(json.dumps({'tracks': [
        {'video_id': 99, 'category_id': {'cannot_decode': True}, 'track_id': {'cannot_decode': True}},
        {'video_id': 10, 'track_id': 1, 'category_id': 3},
        {'video_id': 10, 'track_id': 2, 'category_id': 7},
        {'video_id': 10, 'track_id': 3, 'category_id': 19}]}))
    return assignment, known


def test_policy_projection_gates_other_roles_and_masks_unknown_not_background(tmp_path):
    a, k = fixture(tmp_path)
    projection, proof = project_policy_loss_targets(a, k, fit_classes=1, pseudo_classes=1,
        videos=1, known_tracks=1, pseudo_tracks=1)
    targets, joined = reconcile_policy_targets([(10, i) for i in (1, 2, 3, 4)], projection,
        physical_tracks=4, masked_tracks=2)
    assert targets == [('Known', 0), ('policy_pseudo', 0),
                       ('outside_eligible_Known', None), ('no_retained_semantic_truth', None)]
    assert not proof['development_final_or_other_video_semantic_values_decoded']
    assert joined['masked_loss_tracks'] == 2
    assert not joined['masked_tracks_dropped_or_declared_background']
    with pytest.raises(ValueError):
        reconcile_policy_targets([(10, i) for i in (1, 2, 4)], projection,
                                 physical_tracks=3, masked_tracks=1)
    with pytest.raises(ValueError):
        project_policy_loss_targets(a, k, fit_classes=1, pseudo_classes=1,
                                    videos=1, known_tracks=2, pseudo_tracks=1)


def arrays():
    visual = np.zeros((3, 768), np.float16)
    visual[0, 0] = visual[1, 1] = visual[2, 2] = 1
    return {'track_id': np.array([12, 99], np.int64),
            'offsets': np.array([0, 2, 3], np.int64), 'visual': visual,
            'geometry': np.tile([.1, .2, .5, .8], (3, 1)).astype(np.float32),
            'quality': np.ones(3, np.float32),
            'frame_index': np.array([1, 5, 1], np.int64),
            'clock_frame_index': np.array([0, 1, 3, 5], np.int64)}


def run(a, routes=None):
    router = CurrentObservationRouter(); result = []
    for frame, observations in policy_current_frames(a, routing_bindings=routes):
        batch = router.ingest(frame, observations)
        result.append((frame, tuple(visible_bytes(v) for v in batch.model_views)))
    return result


def test_full_metadata_clock_empty_frames_singletons_renaming_and_future_mutation():
    a = arrays(); baseline = run(a)
    assert [f for f, _ in baseline] == [0, 1, 3, 5]
    assert not baseline[0][1] and not baseline[2][1]
    assert len(baseline[1][1]) == 2 and len(baseline[3][1]) == 1
    assert run(a, {12: 'renamed-z', 99: 'not-a-number'}) == baseline
    mutated = {k: v.copy() for k, v in a.items()}
    mutated['visual'][1] = 0; mutated['visual'][1, 7] = 1
    alternative = run(mutated)
    assert alternative[:3] == baseline[:3] and alternative[3] != baseline[3]
    frames = list(policy_current_frames(a))
    a['visual'][:] = 0
    assert np.linalg.norm(frames[1][1][0].visual) == 1
    assert {f.name for f in fields(CurrentObservation)} == {
        'routing_id', 'visual', 'box_normalized_xyxy', 'quality'}


def test_bad_external_routing_refuses_without_trimming_physical_tracks():
    with pytest.raises(ValueError):
        list(policy_current_frames(arrays(), routing_bindings={12: 'same', 99: 'same'}))
    with pytest.raises(ValueError):
        list(policy_current_frames(arrays(), routing_bindings={12: 'one'}))
