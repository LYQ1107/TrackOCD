import json
from pathlib import Path
import pytest
from src.trackocd_v2.io import sha256_file
from src.trackocd_cvpr.spread_C_development_complete_matrix import verified_matrix,aggregate_matrix,trial_sources

ROOT = Path(__file__).resolve().parents[2]
REPORT = 'outputs/trackocd_cvpr/cvpr3/paired_spread_Known_rejection_all6_full_dev_matrix_v1.json'


def test_actual_complete_six_endpoint_matrix_includes_third_repair_tradeoff_not_seed_selection():
    paths = trial_sources('Known_set_rejection',1029)
    hashes = ('b99a2c17ff32fa626f60c77f4f7e4cc8d75860375abe8500bfb0befdc22dea27',
              'acf9acceab60caaae74d475bb760e6f0afcb0c51ab3bc8aea8c25c511ddf2635',
              '7c65a728e29677064bf63c599758b2bdf3c14c2ab94427ab78aa6fb104d20ab2')
    assert all(sha256_file(ROOT/p) == h for p,h in zip(paths,hashes))
    table = verified_matrix(ROOT); assert len(table) == 6
    r = table[('Known_set_rejection',1029)]
    assert r['actual_future_probe'] == {'future_observations_mutated':39,'verified_same_prior_events':True}
    for o,row in r['orders'].items():
        end = row['prefix_snapshots']['all_available']
        assert end['Known_micro_accuracy'] == 220/302
        assert end['declared_pseudo_absorbed_as_Known_queries'] == 75
        assert end['declared_pseudo_anonymous_commit_queries'] == 44
        assert end['certified_cross_video_reuse_micro_lower_bound'] == 0
        assert end['certified_cross_video_reuse_macro_lower_bound'] == 0
        assert end['all_physical_commit_track_coverage'] == 1
        assert row['physical_actions']['EXISTING'] == {'1027':3180,'1028':3188,'1029':3186}[o]
        assert row['strict_route_reassignment_history']['certified_pseudo_history'] == 100
    for arm in ('baseline','Known_set_rejection'):
        for seed in (1027,1028,1029):
            for order in ('1027','1028','1029'):
                assert table[(arm,seed)]['orders'][order]['private_video_ordinal_order_sha256'] == table[('baseline',1027)]['orders'][order]['private_video_ordinal_order_sha256']
    with pytest.raises(ValueError,match='ALL6'):
        aggregate_matrix({k:v for k,v in table.items() if k != ('Known_set_rejection',1029)})


def test_actual_matrix_public_summary_exact_all6_means_preserves_negative_result_and_goal_scope():
    report = json.loads((ROOT/REPORT).read_text()); table = verified_matrix(ROOT); actual = aggregate_matrix(table)
    assert {k:report[k] for k in actual} == actual
    assert report['actual_endpoint_count'] == 6 and report['all6_actual_normal_complete_trials_verified']
    assert report['same_order_ordinal_permutation_across_all6_verified']
    assert len(report['input_sha256']) == 18
    for p,h in report['input_sha256'].items(): assert sha256_file(ROOT/p) == h
    assert not report['new_private_GT_data_or_weights_read'] and report['new_optimizer_steps'] == 0
    assert not report['selected_model_seed_order_or_prefix']
    assert not report['order_replays_are_independent_training_seeds']
    assert not report['calibrated_Raw_coverage_latency_Pred_joint_final_or_Goal_accepted']
    b = report['arms']['baseline']['three_training_seed_mean']['all_available']
    r = report['arms']['Known_set_rejection']['three_training_seed_mean']['all_available']
    delta = report['paired_repair_minus_baseline_three_seed_mean']['all_available']
    assert b['Known_micro_accuracy'] == 282/302
    assert r['Known_micro_accuracy'] < b['Known_micro_accuracy'] and delta['Known_micro_accuracy'] < 0
    assert r['certified_cross_video_reuse_micro_lower_bound'] == b['certified_cross_video_reuse_micro_lower_bound'] == 0
    assert r['all_physical_commit_track_coverage'] == b['all_physical_commit_track_coverage'] == 1
    assert r['declared_pseudo_anonymous_commit_queries'] == 44/3
    assert report['resources']['training_GPU_hours'] == 0
