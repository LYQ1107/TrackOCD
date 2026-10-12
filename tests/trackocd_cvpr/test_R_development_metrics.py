import json
import numpy as np
import pytest
from pathlib import Path
from src.trackocd_cvpr.R_development_metrics import project_dev_labels, reconcile_dev_labels, certified_cross_video_retrieval


def test_byte_project_dev_only_skips_final_policy_labels_and_retains_unknown(tmp_path):
    assignment = tmp_path / 'a.json'; known = tmp_path / 'k.json'
    assignment.write_text(json.dumps({'assignment': {'classes': {'fit_known': [3], 'development_pseudo': [7],
        'independent_final_pseudo': {'must_not_decode': True}, 'policy_train_pseudo': {'must_not_decode': True}},
        'video_ids': {'development_pseudo': [10], 'independent_final_pseudo': {'must_not_decode': True}}}}))
    known.write_text(json.dumps({'tracks': [{'video_id': 99, 'category_id': {'must_not_decode': True}},
        {'video_id': 10, 'track_id': 1, 'category_id': 3}, {'video_id': 10, 'track_id': 2, 'category_id': 7},
        {'video_id': 10, 'track_id': 3, 'category_id': 19}]}))
    labels, _ = project_dev_labels(assignment, known, expected_fit_classes=1, expected_dev_classes=1,
                                  expected_videos=1, expected_known_tracks=1, expected_pseudo_tracks=1)
    rows, counts = reconcile_dev_labels([(10, i) for i in (1, 2, 3, 4)], labels,
                                        expected_physical=4, expected_unlabeled=1, expected_outside=1)
    assert rows[-1] == ('no_retained_semantic_truth', None) and rows[-2] == ('outside_eligible_Known', None)
    assert counts['declared_pseudoNovel'] == 1
    with pytest.raises(ValueError): reconcile_dev_labels([(10, 1)], labels)


def test_all_gallery_ties_unknown_and_no_positive_queries_not_dropped_or_ID_sorted():
    e = np.zeros((5, 768)); e[:, 0] = 1
    labels = [('declared_pseudoNovel', 7), ('declared_pseudoNovel', 7), ('declared_pseudoNovel', 8),
              ('Known', 3), ('no_retained_semantic_truth', None)]
    result = certified_cross_video_retrieval(e, [1, 2, 3, 4, 5], labels)
    assert result['declared_query_tracks'] == 3 and result['physical_gallery_tracks'] == 5
    assert result['queries_without_verified_cross_video_positive_retained'] == 1
    assert result['queries_with_semantic_unknown_in_top_tie'] == 3
    assert result['certified_lower_bound_micro_R1'] == pytest.approx(1 / 6)
    assert result['certified_lower_bound_micro_R5'] == pytest.approx(2 / 3)
    order = [4, 2, 0, 3, 1]
    permuted = certified_cross_video_retrieval(e[order], [5, 3, 1, 4, 2], [labels[i] for i in order])
    assert permuted == result


def test_same_video_not_a_positive_and_perfect_other_video_certified():
    e = np.zeros((3, 768)); e[0, 0] = e[1, 0] = e[2, 1] = 1
    labels = [('declared_pseudoNovel', 7)] * 2 + [('Known', 3)]
    same = certified_cross_video_retrieval(e, [1, 1, 2], labels)
    assert same['queries_without_verified_cross_video_positive_retained'] == 2
    assert same['certified_lower_bound_micro_R1'] == 0
    cross = certified_cross_video_retrieval(e, [1, 2, 3], labels)
    assert cross['certified_lower_bound_micro_R1'] == 1


def test_frozen_metric_denominators_and_no_checkpoint_or_final_selection():
    root = Path(__file__).resolve().parents[2]
    c = json.loads((root / 'configs/trackocd_cvpr/R_GT_development_evaluation_v1.json').read_text())
    assert c['physical_tracks'] == 302 + 119 + 2 + 49 == 472
    assert c['prefixes'] == [1, 2, 4, 8, 16, 'all_available']
    assert not c['independent_final_Val_Test_model_selection']
    assert not c['automatic_evaluation_training_or_Goal_acceptance']
