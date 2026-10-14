import json
from pathlib import Path
import pytest
from src.trackocd_cvpr import arbitration_development_comparison as comparison
from tests.trackocd_cvpr.test_arbitration_development_receipts import forbid_private_stat

ROOT = Path(__file__).resolve().parents[2]


def test_actual_all5_complete_public_evidence_preserves_each_order_and_adverse_shadow(monkeypatch):
    forbid_private_stat(monkeypatch)
    r = comparison.complete_comparison(ROOT)
    assert json.loads((ROOT/'outputs/trackocd_cvpr/cvpr3/arbitration_all5_full_development_comparison_v1.json').read_text()) == r
    assert r['qualification']['all5_actual_complete_Train_dev_attribution_verified']
    assert set(r['models']) == {'initial_Raw','calendar1027','calendar1028','calendar1029','ONE_fullpolicy'}
    for row in r['models'].values():
        assert set(row['full_orders']) == {'1027','1028','1029'}
        for order in row['full_orders'].values():
            assert order['full_clock']['canonical_groups'] == 8753
            assert sum(s['physical_observations'] for s in order['external_strata'].values()) == 8753
            assert order['external_strata']['masked'].get('same_history_remove_own_correct_Known_groups',0) == 0
    raw = r['models']['initial_Raw']['mean_counts_over3orders_not_independent_trials']
    one = r['models']['ONE_fullpolicy']['mean_counts_over3orders_not_independent_trials']
    assert raw['Known_correct_one_step_shadow_count'] == 116
    assert raw['declared_pseudo_Known_one_step_shadow_count'] == 34
    assert one['Known_correct_one_step_shadow_count'] == 163
    assert one['declared_pseudo_Known_one_step_shadow_count'] == pytest.approx(238/3)
    assert r['three_calendar_seed_mean_over_all3orders']['Known_correct_one_step_shadow_count'] == pytest.approx(1131/9)
    assert r['three_calendar_seed_mean_over_all3orders']['declared_pseudo_Known_one_step_shadow_count'] == pytest.approx(735/9)
    assert not r['private_GT_weights_rows_loaded']
    assert not r['conditional_shadow_is_full_intervention_gain']
    assert not r['model_seed_threshold_or_coefficient_selected']
    assert not r['joint_Novel_R_C_M_Pred_dense_calibrated_controls_final_or_Goal_accepted']


def test_partial_matrix_refused_before_comparison_input_reads(monkeypatch,tmp_path):
    def refuse(root):raise ValueError('Complete original full-dev result/launch/normal terminal required')
    def forbid_read(*args,**kwargs):raise AssertionError('Comparison read before all5 qualification')
    monkeypatch.setattr(comparison.receipts,'verified_matrix',refuse)
    monkeypatch.setattr(Path,'read_text',forbid_read)
    with pytest.raises(ValueError,match='Complete original full-dev'):
        comparison.complete_comparison(tmp_path)
