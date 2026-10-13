import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_actual_M_third_seed_first_control_fixed200_independent_initialization_and_dirty_history():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_GT_window_v1_relative_quarantine_s1029'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '06d60507428e03b97c6f05c6d46661f563e6ecf7514582f89744e1260e663bbc'
    r = json.loads((out / 'result.json').read_text()); s = r['summary']
    assert (r['variant'], r['seed'], s['optimizer_steps'], s['trainable_parameters']) == ('relative_quarantine', 1029, 200, 161)
    assert s['model_parameter_sha256_before'] == '7daf96dfb173f62b6d797cd63f5d09dda7f0660154d487b27250430eb50161e3'
    for seed in (1027, 1028):
        previous = json.loads((out.with_name('M_GT_window_v1_relative_quarantine_s' + str(seed)) / 'result.json').read_text())
        assert s['model_parameter_sha256_before'] != previous['summary']['model_parameter_sha256_before']
    assert s['actual_optimizer_family_groups'] == {'Known': 577, 'policy_pseudo': 113}
    assert s['actual_optimizer_policy_pseudo_classes'] == 2
    assert s['actual_optimizer_target_bins'] == {'fractional': 20, 'one': 432, 'zero': 238}
    assert r['endpoint'] == {'bytes': 12515, 'sha256': '08891915c65ceaa6b6ed1b025cc3e48f90e1f1bbedb2d51f088331a63f780941', 'reopen_exact': True}
    for e in s['episodes']:
        assert (e['videos'], e['metadata_frames'], e['physical_observations'], e['empty_frames'], e['optimizer_steps']) == (10, 374, 1268, 1, 50)
        assert e['actual_anonymous_commits_retained'] == e['actual_bank_commits_retained'] == 1268
        assert e['external_strict_version_lineage']['all_parent_history_and_empty_tombstones_retained']
    final = s['episodes'][-1]['external_strict_version_lineage']['strict_latest_version_status_counts']
    assert final['VERIFIED_BAD_ANONYMOUS_KNOWN_HISTORY'] == 31
    assert final['VERIFIED_BAD_MIXED_PSEUDO_HISTORY'] == 1
    assert not r['three_seed_M_strict_CT_Pred_or_Goal_accepted'] and not r['dev_final_Val_Test_or_new_nonKnown_labels_used']
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
    assert t['last_persisted_completed_optimizer_steps'] == 200
