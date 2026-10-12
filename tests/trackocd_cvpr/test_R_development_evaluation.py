import numpy as np
import pytest
import json
from pathlib import Path
from src.trackocd_cvpr.R_development_evaluation import effective_rank, known_recognition, embed_complete_role
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
from tests.trackocd_cvpr.test_fit_R_training import track


def test_full_role_prefixes_singleton_and_raw_identity_no_future_aggregation():
    rows = [track(1, None, 1), track(2, None, 2)]
    raw, adapted, actual = embed_complete_role(RawPreservingEvidence(), rows, 1, lambda: None)
    assert actual == [1, 1] and raw.shape == adapted.shape == (2, 768)
    np.testing.assert_array_equal(raw, adapted)
    _, _, actual = embed_complete_role(RawPreservingEvidence(), rows, 'all_available', lambda: None)
    assert actual == [3, 3]


def test_rank_and_known_exact_ties_without_GT_prototype_repair():
    e = np.zeros((3, 768)); e[0, 0] = e[1, 0] = e[2, 1] = 1
    assert effective_rank(e) > 1
    labels = [('Known', 7), ('Known', 8), ('no_retained_semantic_truth', None)]
    p = np.zeros((2, 768)); p[:, 0] = 1
    result = known_recognition(e, labels, {7: 0, 8: 1}, p)
    assert result['Known_queries'] == 2 and result['micro_accuracy_exact_tie_expectation'] == 0.5
    with pytest.raises(ValueError): known_recognition(e, labels, {7: 0}, p)


def test_actual_evaluator_requires_registered_complete_cache_and_fixedendpoint_not_winner():
    root = Path(__file__).resolve().parents[2]
    c = json.loads((root / 'configs/trackocd_cvpr/R_development_first_pair_v1.json').read_text())
    assert c['models'] == ['temporal_Raw_teacher', 'timeless_Raw_teacher']
    assert c['seed'] == 1027 and c['checkpoint_step'] == 200
    assert c['prefixes'] == [1, 2, 4, 8, 16, 'all_available']
    assert not c['select_model_or_checkpoint']
    assert c['policy_dev_execution_commit'] == 'd3eaf6035ac1deaa92582bf46fe6966f88e70032'
