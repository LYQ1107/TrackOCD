import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_actual_complete_policy_exclusion_controls_keep_full_orders_probes_views_and_denominators():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/self_excluded_Mean_policy_pilot_v1'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '9987e002ece8e906fe4ad85750c8f08f622dee8253011587b78dfe996b9df914'
    r = json.loads((out / 'result.json').read_text())
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert r['live_remote_registration_proof']['sha'] == '50132e64e208119962383c7b0744f4443b8dff09'
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
    assert t['complete_result_present'] and t['wall_seconds_including_registration'] < 3000
    assert set(r['controls']) == {'Mean_latest', 'exclude_query_alias_scope', 'exclude_current_video'}
    for name, known_correct in (('Mean_latest', 0), ('exclude_query_alias_scope', 2), ('exclude_current_video', 3)):
        control = r['controls'][name]
        assert control['actual_complete_OLD_cache_shadow_equal']
        assert control['actual_full_ID_rename_reverse_event_ledger_query_view_and_audit_equal']
        assert control['actual_future_observations_mutated'] == 141
        assert control['prior_events_versions_ledger_and_filtered_query_views_invariant']
        assert control['combined_derived_cache_bytes'] <= 67108864
        assert set(control['orders']) == {'1027', '1028', '1029'}
        for order, row in control['orders'].items():
            assert (row['full_physical_bitset_scope'], row['observations'], row['metadata_frames'], row['empty_frames']) == (54, 1268, 374, 1)
            assert sum(row['physical_actions'].values()) == 1268
            assert set(row['chosen_preframe_version_prefix_snapshots']) == {'1', '2', '4', '8', '16', 'all_available'}
            end = row['chosen_preframe_version_prefix_snapshots']['all_available']
            assert end['Known_micro_accuracy'] == known_correct / 34
            reuse = 1 if name == 'exclude_current_video' and order == '1028' else 0
            assert end['certified_cross_video_reuse_micro_lower_bound'] == reuse / 10
            assert end['all_physical_commit_track_coverage'] == 1
            if name != 'Mean_latest':
                v = row['actual_query_view_counts']
                assert v['query_views'] == 1268 and v['retained_group_opportunities'] < v['original_group_opportunities']
                assert v['views_with_excluded_groups'] == (1214 if name == 'exclude_query_alias_scope' else 1231)
    assert not r['video_orders_are_independent_training_seeds']
    assert not r['strict_history_certificate_is_filtered_active_support_semantic_independence']
    assert not r['full_development_Pred_joint_or_Goal_accepted']
    assert r['new_optimizer_steps'] == 0 and r['resources']['training_GPU_hours'] == 0 and not r['resources']['CUDA_initialized']
