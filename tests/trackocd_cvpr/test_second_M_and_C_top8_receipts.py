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


def test_M_seed1028_noquarantine_fit_diverges_from_pair_not_matched_dev_gain():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_GT_window_v1_relative_weighted_s1028'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '38de22b4ac2ee236408e799383598cbad05e4a2acb71056eef9f43cc58fea705'
    r = json.loads((out / 'result.json').read_text()); s = r['summary']
    pair = json.loads((out.with_name('M_GT_window_v1_relative_quarantine_s1028') / 'result.json').read_text())
    assert s['model_parameter_sha256_before'] == pair['summary']['model_parameter_sha256_before']
    assert s['model_parameter_sha256_after'] != pair['summary']['model_parameter_sha256_after']
    assert (s['optimizer_steps'], s['trainable_parameters']) == (200, 161)
    assert s['actual_optimizer_family_groups'] == {'Known': 474, 'policy_pseudo': 136}
    assert s['actual_optimizer_policy_pseudo_classes'] == 2
    assert s['actual_optimizer_target_bins'] == {'one': 338, 'zero': 272}
    assert r['endpoint'] == {'bytes': 12515, 'sha256': 'c4c04d2dc7945ae4e597c80e872377ec989879113c1dfd029218ea163a9b5ab0', 'reopen_exact': True}
    for ep in s['episodes']:
        assert (ep['videos'], ep['metadata_frames'], ep['physical_observations'], ep['empty_frames'], ep['optimizer_steps']) == (10, 374, 1268, 1, 50)
        assert ep['actual_anonymous_commits_retained'] == ep['actual_bank_commits_retained'] == 1268
        assert ep['version_lifecycle_records'].get('Quarantined', 0) == 0
    assert not r['three_seed_M_strict_CT_Pred_or_Goal_accepted']
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
    assert t['last_persisted_completed_optimizer_steps'] == 200


def test_C_first_seed_pooled_control_full3orders_no_certified_reuse():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Top8_pool_context_s1027'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '596611d0a17c563621bf118b72a294cbfe03fd884a5387fbed611ec92273ca9c'
    r = json.loads((out / 'result.json').read_text())
    assert r['model'] == 'Top8_pool_context' and r['training_seed'] == 1027
    assert r['actual_full_ID_rename_reverse_stream_equal'] and r['actual_future_probe']['future_observations_mutated'] == 39
    for order, known, absorbed, anon in (('1027', 280, 114, 5), ('1028', 282, 116, 3), ('1029', 282, 117, 2)):
        row = r['orders'][order]; end = row['prefix_snapshots']['all_available']
        assert (row['metadata_frames'], row['observations'], row['empty_frames']) == (2562, 8753, 116)
        assert end['physical_tracks'] == 472 and end['declared_pseudo_queries'] == 119
        assert end['Known_micro_accuracy'] == pytest.approx(known / 302)
        assert end['declared_pseudo_absorbed_as_Known_queries'] == absorbed
        assert end['declared_pseudo_anonymous_commit_queries'] == anon
        assert end['certified_cross_video_reuse_micro_lower_bound'] == 0
        assert end['all_physical_commit_track_coverage'] == 1
    assert not r['model_or_checkpoint_selection'] and not r['full_semantic_truth_Hscore_CT_M_Pred_or_Goal_accepted']
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None


def test_C_second_independent_Top1_seed_repeats_actual_closed_set_collapse():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Top1_conditioned_s1028'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '6cfe231185e2ce5bb4f5db02aff58fc1ee11b4207147abbb990f526c6431baba'
    r = json.loads((out / 'result.json').read_text())
    assert r['model'] == 'Top1_conditioned' and r['training_seed'] == 1028
    assert r['actual_full_ID_rename_reverse_stream_equal'] and r['actual_future_probe']['verified_same_prior_events']
    for row in r['orders'].values():
        end = row['prefix_snapshots']['all_available']
        assert row['physical_actions'] == {'KNOWN': 8753}
        assert end['Known_micro_accuracy'] == pytest.approx(282 / 302)
        assert end['declared_pseudo_absorbed_as_Known_queries'] == 119
        assert end['declared_pseudo_anonymous_commit_queries'] == 0
        assert end['certified_cross_video_reuse_micro_lower_bound'] == 0
        assert end['physical_tracks'] == 472 and end['all_physical_commit_track_coverage'] == 1
    assert not r['model_or_checkpoint_selection'] and not r['full_semantic_truth_Hscore_CT_M_Pred_or_Goal_accepted']
    terminal = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0 and terminal['guard_stop_reason'] is None
