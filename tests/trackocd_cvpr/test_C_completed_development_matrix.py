import hashlib
import json
from pathlib import Path
import pytest
from src.trackocd_cvpr.C_development_complete_matrix import verified_matrix, aggregate_matrix

ROOT = Path(__file__).resolve().parents[2]


def test_actual_last_C_endpoint_and_complete_public_all9_matrix_normal_plus_exact_exception():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Top8_pool_context_s1029'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == 'f334c220d4922d723297d9112aa1ef288f19c5f6de206e259446cb8b03e0c068'
    r = json.loads((out / 'result.json').read_text())
    for order, known, reused, absorbed in (('1027', 173, 6, 46), ('1028', 164, 8, 33), ('1029', 175, 4, 48)):
        end = r['orders'][order]['prefix_snapshots']['all_available']
        assert end['Known_micro_accuracy'] == pytest.approx(known / 302)
        assert end['certified_cross_video_reuse_micro_lower_bound'] == pytest.approx(reused / 119)
        assert end['declared_pseudo_absorbed_as_Known_queries'] == absorbed
        assert end['declared_pseudo_anonymous_commit_queries'] == 119 - absorbed
    table = verified_matrix(ROOT)
    assert len(table) == 9
    result = aggregate_matrix(table)
    assert not result['selected_model_seed_order_or_prefix']
    assert not result['order_replays_are_independent_training_seeds']
    assert not result['official_CT_full_semantic_truth_Pred_or_Goal_accepted']
