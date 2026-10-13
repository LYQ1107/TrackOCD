import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_actual_M_second_seed_triplet_complete_transport_failure_not_erased():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_GT_window_v1_pooled_weighted_s1028_transport_retry1'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '30f993828d051cdfff8c9d228786bf4304951f84cb5d789cc0343b238c47d908'
    r = json.loads((out / 'result.json').read_text()); s = r['summary']
    controls = [json.loads((out.with_name('M_GT_window_v1_' + v + '_s1028') / 'result.json').read_text()) for v in ('relative_quarantine', 'relative_weighted')]
    assert (r['variant'], r['seed'], s['optimizer_steps'], s['trainable_parameters']) == ('pooled_weighted', 1028, 200, 161)
    assert all(s['model_parameter_sha256_before'] == row['summary']['model_parameter_sha256_before'] for row in controls)
    assert all(s['model_parameter_sha256_after'] != row['summary']['model_parameter_sha256_after'] for row in controls)
    assert s['actual_optimizer_family_groups'] == {'Known': 474, 'policy_pseudo': 136}
    assert s['actual_optimizer_policy_pseudo_classes'] == 2
    assert r['endpoint'] == {'bytes': 12515, 'sha256': '18391631ab6679c7ec7da0b21bac6e9a560ec54fd5fc2acf99c35ab775eb8158', 'reopen_exact': True}
    for e in s['episodes']:
        assert (e['videos'], e['metadata_frames'], e['physical_observations'], e['empty_frames'], e['optimizer_steps']) == (10, 374, 1268, 1, 50)
        assert e['actual_anonymous_commits_retained'] == e['actual_bank_commits_retained'] == 1268
        assert not e['version_lifecycle_records'].get('Quarantined', 0)
    assert not r['three_seed_M_strict_CT_Pred_or_Goal_accepted']
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None and t['last_persisted_completed_optimizer_steps'] == 200
    old = out.with_name('M_GT_window_v1_pooled_weighted_s1028_supervisor_attempt_1.json')
    assert hashlib.sha256(old.read_bytes()).hexdigest() == '311856168731045c5e6ac46ab4f5d5ac32fe0c59cb2268c59c3160f06ffb5afc'
    assert not json.loads(old.read_text())['successful_owned_execution']


def test_C_second_Top8_seed_actual_closed_set_failure_not_missing_pseudo_optimizer_loss():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Top8_conditioned_s1028'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '080011a3d8edb127cb7c4615645bcd5b6be6abc2e264bb19356583b69e1c13a2'
    r = json.loads((out / 'result.json').read_text())
    assert r['model'] == 'Top8_conditioned' and r['training_seed'] == 1028
    assert r['actual_full_ID_rename_reverse_stream_equal'] and r['actual_future_probe'] == {'future_observations_mutated': 39, 'verified_same_prior_events': True}
    for row in r['orders'].values():
        assert row['physical_actions'] == {'KNOWN': 8753}
        end = row['prefix_snapshots']['all_available']
        assert end['Known_micro_accuracy'] == pytest.approx(282 / 302)
        assert end['declared_pseudo_absorbed_as_Known_queries'] == 119 and end['declared_pseudo_anonymous_commit_queries'] == 0
        assert end['certified_cross_video_reuse_micro_lower_bound'] == 0
        assert end['physical_tracks'] == 472 and end['all_physical_commit_track_coverage'] == 1
    fit = json.loads((out.with_name('C_GT_window_v1_Top8_conditioned_s1028') / 'result.json').read_text())
    assert fit['summary']['actual_optimizer_policy_pseudo_classes'] == 2
    assert fit['summary']['actual_optimizer_family_groups'] == {'Known': 381, 'policy_pseudo': 97}
    assert not r['model_or_checkpoint_selection'] and not r['full_semantic_truth_Hscore_CT_M_Pred_or_Goal_accepted']
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
