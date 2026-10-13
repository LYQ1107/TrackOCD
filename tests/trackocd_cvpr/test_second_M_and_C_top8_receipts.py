"""Public immutable receipts: genuine execution does not imply novelty gain."""
import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_second_M_seed_actual_fixed_trial_with_quarantine_not_purity_acceptance():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_GT_window_v1_relative_quarantine_s1028'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '7ead572e684de3ad448cf353e2df30b2deb3e01d6096df3c4bb22be471236d1a'
    r = json.loads((out / 'result.json').read_text()); s = r['summary']
    first = json.loads((out.with_name('M_GT_window_v1_relative_quarantine_s1027') / 'result.json').read_text())
    assert (r['variant'], r['seed'], s['optimizer_steps'], s['trainable_parameters']) == ('relative_quarantine', 1028, 200, 161)
    assert s['model_parameter_sha256_before'] != first['summary']['model_parameter_sha256_before']
    assert s['actual_optimizer_family_groups'] == {'Known': 491, 'policy_pseudo': 122}
    assert s['actual_optimizer_policy_pseudo_classes'] == 2
    assert s['actual_optimizer_target_bins'] == {'fractional': 6, 'one': 358, 'zero': 249}
    assert len(s['episodes']) == 4 and len(s['loss_curve']) == 200
    for ep in s['episodes']:
        assert (ep['videos'], ep['metadata_frames'], ep['physical_observations'], ep['empty_frames'], ep['optimizer_steps']) == (10, 374, 1268, 1, 50)
        assert ep['actual_anonymous_commits_retained'] == ep['actual_bank_commits_retained']
        assert ep['external_strict_version_lineage']['all_parent_history_and_empty_tombstones_retained']
    assert [ep['version_lifecycle_records'].get('Quarantined', 0) for ep in s['episodes']] == [0, 184, 824, 0]
    assert r['endpoint'] == {'bytes': 12515, 'sha256': '316db258f372e4ebcfe802200efb69c7854ebe7e1477cd46f702b74a9b532744', 'reopen_exact': True}
    assert not r['three_seed_M_strict_CT_Pred_or_Goal_accepted']
    assert not r['dev_final_Val_Test_or_new_nonKnown_labels_used']
    terminal = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0 and terminal['guard_stop_reason'] is None
    assert terminal['last_persisted_completed_optimizer_steps'] == 200


def test_actual_C_top8_seed1027_still_absorbs_most_pseudoNovel_no_reuse_gain():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Top8_conditioned_s1027'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '06323d59e5d605f31e52551a8681b9f5c79374c6f7498e8ebec8dd683e87388e'
    r = json.loads((out / 'result.json').read_text())
    assert r['model'] == 'Top8_conditioned' and r['training_seed'] == 1027
    assert r['optimizer_steps'] == 0 and not r['model_or_checkpoint_selection']
    assert r['actual_full_ID_rename_reverse_stream_equal']
    assert r['actual_future_probe'] == {'future_observations_mutated': 39, 'verified_same_prior_events': True}
    assert set(r['orders']) == {'1027', '1028', '1029'}
    for row in r['orders'].values():
        assert (row['metadata_frames'], row['observations'], row['empty_frames']) == (2562, 8753, 116)
        assert row['physical_actions'] == {'EXISTING': 131, 'KNOWN': 8614, 'NEW': 8}
        end = row['prefix_snapshots']['all_available']
        assert (end['physical_tracks'], end['Known_queries'], end['declared_pseudo_queries']) == (472, 302, 119)
        assert end['Known_micro_accuracy'] == pytest.approx(282 / 302)
        assert end['declared_pseudo_absorbed_as_Known_queries'] == 117
        assert end['declared_pseudo_anonymous_commit_queries'] == 2
        assert end['certified_cross_video_reuse_micro_lower_bound'] == 0
        assert end['all_physical_commit_track_coverage'] == 1
    assert not r['full_semantic_truth_Hscore_CT_M_Pred_or_Goal_accepted']
    terminal = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0 and terminal['guard_stop_reason'] is None
