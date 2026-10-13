import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_actual_complete_state_EMA_retains_full_denominators_probes_profile_and_normal_completion():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_fingerprint_cache_v1_StateEMA_latest'
    digest = 'a17babc248ec10472e3ed5d2eac61ca6beedc70f1c864b488352bece5eb76a1b'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == digest
    assert hashlib.sha256((out / 'execution_profile.json').read_bytes()).hexdigest() == '01d9b3a200196bd40002100eab1a9424c1c47b0d9c2e0d8e7d7f0f55762d5924'
    r = json.loads((out / 'result.json').read_text()); p = json.loads((out / 'execution_profile.json').read_text())
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert r['model'] == 'StateEMA_latest' and r['training_seed'] is None and r['optimizer_steps'] == 0
    assert r['live_remote_registration_proof']['sha'] == '807ccd5e37d5e59a305ae8a4ec5d186b1aa311f0'
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
    assert t['complete_result_present'] and t['wall_seconds_including_imports_and_network'] < 3000
    assert t['peak_sampled_worker_threads'] == 1 and t['peak_sampled_owned_worker_RSS_bytes'] < 3221225472
    assert r['actual_full_ID_rename_reverse_stream_equal']
    assert r['actual_future_probe'] == {'future_observations_mutated': 39,
        'verified_same_prior_events_chosen_versions_and_numeric_ledger': True}
    assert p['result_sha256'] == digest and p['actual_entire_ALL11_fingerprint_shadow_profile_checked']
    assert p['Raw_statistics_bytes'] + p['fingerprint_bytes'] <= 67108864
    assert p['fingerprint_hits'] == 3404357 and p['fingerprint_misses'] == 35873
    assert set(r['orders']) == {'1027', '1028', '1029'}
    for order, known, reuse, polluted in (('1027', 22, 2, 176), ('1028', 20, 6, 175), ('1029', 15, 14, 167)):
        row = r['orders'][order]
        assert (row['full_physical_bitset_scope'], row['observations'], row['metadata_frames'], row['empty_frames']) == (472, 8753, 2562, 116)
        assert sum(row['physical_actions'].values()) == 8753
        assert row['strict_latest_version_status_counts']['VERIFIED_BAD_ANONYMOUS_KNOWN_HISTORY'] == polluted
        snapshots = row['chosen_preframe_version_prefix_snapshots']
        assert set(snapshots) == {'1', '2', '4', '8', '16', 'all_available'}
        assert snapshots['all_available']['Known_micro_accuracy'] == known / 302
        assert snapshots['all_available']['certified_cross_video_reuse_micro_lower_bound'] == reuse / 119
    assert not r['current_active_purity_official_CT_self_exclusion_Pred_joint_or_Goal_accepted']
    assert not r['final_Val_Test_or_new_nonKnown_semantic_values_read']
    assert not r['resources']['CUDA_initialized'] and r['resources']['training_GPU_hours'] == 0
