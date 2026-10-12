"""First actual C optimization is plumbing, not persistent discovery success."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr3/candidate_optimizer_pilot_v1'


def test_actual_owned_C_updates_full_physical_rollout_and_unfixed_zero_reuse():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '71f664b63f1a035ca034d8430d6db6b8269896f0470863e22b559519fac0a8a8'
    r = json.loads((OUT / 'result.json').read_text()); summary = r['summary']; rollout = summary['complete_rollout']
    assert r['live_remote_registration_proof']['sha'] == '1be024c331dbe10aafd016bda110a818c97e8941'
    assert summary['optimizer_steps'] == 10 and summary['trainable_parameters'] == 708
    assert [x['step'] for x in summary['loss_curve']] == list(range(1, 11))
    assert all(x['gradient_norm_before_clip'] > 0 for x in summary['loss_curve'])
    assert (rollout['videos'], rollout['metadata_frames'], rollout['empty_frames'], rollout['physical_observations']) == (10, 374, 1, 1268)
    assert rollout['physical_actions'] == {'NEW': 655, 'KNOWN': 613}
    assert rollout['physical_actions'].get('EXISTING', 0) == 0  # Do not hide uncorrected failure.
    assert sum(summary['loss_plan_counts_all_frames'].values()) == 1268
    assert summary['loss_plan_counts_all_frames']['MASKED_QUERY_OR_CONFLICTING_ALIASES'] == 271
    assert summary['actual_anonymous_commits_retained'] == summary['actual_bank_commits_retained'] == 655
    assert summary['allocated_tokens'] == 655
    assert not summary['GT_created_reassigned_or_repaired_bank_or_removed_wrong_history']
    assert not summary['self_support_exclusion_or_independent_category_reuse_or_C_M_scientific_acceptance']
    assert r['policy_scope']['physical_tracks'] == 54 and r['policy_scope']['masked_loss_tracks'] == 10
    assert r['offline_known_prior_and_policy_loss_binding_order_exact']
    assert r['endpoint']['reopen_exact'] and r['endpoint']['bytes'] == 23975
    assert not r['dev_final_Val_Test_or_new_nonKnown_labels_used']
    assert not r['all3seeds_or_distinct_R_C_M_or_qualified_Pred_or_Goal_accepted']
    terminal = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None and not r['resources']['CUDA_initialized']
