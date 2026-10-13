import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_GT_window_v1_relative_quarantine_s1027'


def test_first_real_M_fixed200_receipt_has_full_trace_strict_audit_and_exposure_limits():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '148aeaa013e2a89f30ab63e63821d083af5bc5c217fbde413c946d0e6332decb'
    r = json.loads((OUT / 'result.json').read_text()); s = r['summary']
    assert r['variant'] == 'relative_quarantine' and r['seed'] == 1027
    assert s['optimizer_steps'] == 200 and s['trainable_parameters'] == 161
    assert s['actual_optimizer_family_groups'] == {'Known': 519, 'policy_pseudo': 89}
    assert s['actual_optimizer_semantic_classes'] == 2 and s['actual_optimizer_policy_pseudo_classes'] == 1
    assert s['actual_optimizer_target_bins'] == {'one': 454, 'zero': 154}
    assert len(s['episodes']) == 4 and len(s['loss_curve']) == 200
    for ep in s['episodes']:
        assert (ep['videos'], ep['metadata_frames'], ep['physical_observations'], ep['empty_frames'], ep['optimizer_steps']) == (10, 374, 1268, 1, 50)
        assert ep['actual_anonymous_commits_retained'] == ep['actual_bank_commits_retained'] == 1268
        assert ep['external_strict_version_lineage']['all_parent_history_and_empty_tombstones_retained']
        assert ep['external_strict_version_lineage']['strict_latest_version_status_counts']['VERIFIED_BAD_ANONYMOUS_KNOWN_HISTORY'] >= 28
        assert ep['version_lifecycle_records'].get('Quarantined', 0) == 0  # no actual quarantine success claimed
    assert r['endpoint'] == {'bytes': 12515, 'sha256': '6674a0fbefd25f9e1e5d12c00c9e5a9c7b6825abe43f14dc739be46983b68729', 'reopen_exact': True}
    assert r['checkpoint_selection'] == 'Fixed200 endpoint' and not r['old_R_C_M_or_pilot_warmstart']
    assert not r['dev_final_Val_Test_or_new_nonKnown_labels_used'] and not r['three_seed_M_strict_CT_Pred_or_Goal_accepted']
    t = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None


def test_actual_second_M_control_has_exact_summary_parameter_and_private_trace_parity():
    out = OUT.with_name('M_GT_window_v1_relative_weighted_s1027')
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == 'ed91a858d0153210a4e76c9c5826274abc4f62bf82aa93384f5542579194a5ef'
    r = json.loads((out / 'result.json').read_text()); full = json.loads((OUT / 'result.json').read_text())
    assert r['variant'] == 'relative_weighted' and r['seed'] == 1027
    assert r['summary'] == full['summary']
    assert r['private_prediction_history'] == full['private_prediction_history']
    assert r['endpoint'] == {'bytes': 12515, 'sha256': '1f90a897056efd38034d522bad6ae2b125d79c02ed735d39485751f43880b086', 'reopen_exact': True}
    assert not r['old_R_C_M_or_pilot_warmstart'] and not r['three_seed_M_strict_CT_Pred_or_Goal_accepted']
    terminal = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None and terminal['last_persisted_completed_optimizer_steps'] == 200

