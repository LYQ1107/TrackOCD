import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.iclr27_phase76r.errata import checkpoint_in_safe_window, correct_teacher_authorization


def test_teacher_erratum_uses_global_rows():
    result = correct_teacher_authorization(
        [{'fold': 0, 'delta_r1': -0.03}, {'fold': 1, 'delta_r1': 0.01}],
        [{'fold': 0, 'delta_r1': 0.0}, {'fold': 1, 'delta_r1': 0.0}],
    )
    assert result['old_result'] is True
    assert result['corrected_result'] is False
    assert result['global_bad_folds'] == [{'fold': 0, 'delta_r1': -0.03}]


def test_safe_window_is_diagnostic_only_predicate():
    row = {
        'global_unsafe': 0, 'legal_unsafe': 0,
        'global_delta_r1': -0.001, 'global_delta_map': -0.001,
        'legal_delta_r1': 0.01, 'legal_delta_map': 0.01,
        'mean_raw_adapt_cosine': 0.99,
    }
    assert checkpoint_in_safe_window(row)
    row['global_unsafe'] = 1
    assert not checkpoint_in_safe_window(row)
