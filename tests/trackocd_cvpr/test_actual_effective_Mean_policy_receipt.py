import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr4/effective_Mean_policy_audit_registration_retry1'


def test_actual_full_effective_audit_normal_wait_binds_unchanged_actor_all_controls_and_orders():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '917d6198b3c06a9cd615a73e2a6050e43714b0f0f107aa74aa4cf2446eb2ac31'
    assert hashlib.sha256((OUT / 'effective_support_result.json').read_bytes()).hexdigest() == 'bd68eb8223bed59db0aa747db7e369620a76de3b9d236397b4a9c744c6736266'
    actor = json.loads((OUT / 'result.json').read_text()); supplemental = json.loads((OUT / 'effective_support_result.json').read_text())
    t = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    prior = json.loads((ROOT / 'outputs/trackocd_cvpr/cvpr4/self_excluded_Mean_policy_pilot_v1/result.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
    assert t['complete_result_present'] and t['only_new_owned_Popen_worker_managed']
    assert t['wall_seconds_including_registration'] <= 3000 and t['peak_sampled_owned_worker_RSS_bytes'] <= 3221225472
    assert supplemental['original_actor_result_sha256'] == hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest()
    assert actor['live_remote_registration_proof']['sha'] == 'b53a25b3299e92ab7b31a82e5ae086d5c0b6b414'
    assert actor['new_optimizer_steps'] == supplemental['new_optimizer_steps'] == 0
    assert not actor['resources']['CUDA_initialized'] and actor['resources']['training_GPU_hours'] == 0
    assert set(actor['controls']) == set(supplemental['controls']) == set(prior['controls'])
    for mode, row in supplemental['controls'].items():
        assert row['actual_full_effective_old_cache_ID_reverse_and_prior_future_equal']
        assert row['original_actor_all_orders_and_private_trace_match_previous_pilot']
        assert actor['controls'][mode]['orders'] == prior['controls'][mode]['orders']
        assert actor['controls'][mode]['private_order1027_trace_sha256'] == prior['controls'][mode]['private_order1027_trace_sha256']
        assert set(row['effective_orders']) == {'1027','1028','1029'}
        for metrics in row['effective_orders'].values():
            assert (metrics['full_physical_scope'],metrics['metadata_frames_observed'],metrics['physical_observations_observed']) == (54,374,1268)
            assert not metrics['original_dirty_history_replaced_or_full_semantic_truth_statistical_independence_official_CT_Pred_or_Goal_accepted']
            assert set(metrics['effective_selected_support_prefix_snapshots']) == {'1','2','4','8','16','all_available'}
            for prefix in metrics['effective_selected_support_prefix_snapshots'].values():
                assert (prefix['physical_tracks'],prefix['Known_queries'],prefix['declared_pseudo_queries']) == (54,34,10)
    assert not supplemental['full_development_Pred_statistical_independence_or_Goal_accepted']


@pytest.mark.parametrize('mode,known,cross,unknown', [
    ('Mean_latest',0,0,[11,11,11]), ('exclude_query_alias_scope',2,0,[16,15,16]),
    ('exclude_current_video',3,2,[1,2,0])])
def test_actual_effective_lower_bounds_are_not_old_history_certificates_or_full_purity(mode, known, cross, unknown):
    s = json.loads((OUT / 'effective_support_result.json').read_text())['controls'][mode]
    actor = json.loads((OUT / 'result.json').read_text())['controls'][mode]
    for index,seed in enumerate(('1027','1028','1029')):
        m = s['effective_orders'][seed]['effective_selected_support_prefix_snapshots']['all_available']
        assert round(m['Known_micro_accuracy']*34) == known
        assert round(m['certified_cross_video_reuse_micro_lower_bound']*10) == cross
        assert m['unscorable_reuse_queries_retained'] == unknown[index]
        if mode == 'exclude_current_video':
            h = actor['orders'][seed]['chosen_preframe_version_prefix_snapshots']['all_available']
            assert round(h['certified_cross_video_reuse_micro_lower_bound']*10) == (0,1,0)[index]
            assert m['declared_pseudo_absorbed_as_Known_queries'] == 1
