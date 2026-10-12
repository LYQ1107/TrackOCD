import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_temporal_Raw_teacher_s1027'


def test_first_real_fixed_endpoint_R_trial_is_not_all_seeds_or_scientific_acceptance():
    p = OUT / 'result.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == '981fc224e844d4531a7e97a3fd9db54a9967b76550a1c93a82efd69c701dfcee'
    r = json.loads(p.read_text())
    assert r['status'] == 'COMPLETED_FIXED_R_FIT_TRIAL_NOT_DEV_VALIDATED_OR_GOAL_ACCEPTED'
    assert r['live_remote_registration_proof']['sha'] == '11dad1895b7f9e098b05a75553422307e9476d84'
    assert r['variant'] == 'temporal_Raw_teacher' and r['seed'] == 1027
    assert r['completed_optimizer_steps'] == 200 and len(r['loss_curve']) == 200
    assert r['model_parameter_sha256_before'] != r['model_parameter_sha256_after']
    assert r['endpoint']['reopen_exact'] and r['checkpoint_selection'] == 'Fixed200 endpoint'
    assert not r['old_fitted_adapter_initialization'] and not r['dev_final_Val_Test_targets_or_pixels_used']
    assert not r['all3seeds_or_distinct_R_C_M_mechanisms_or_Goal_A_B_C_accepted']
    t = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
    assert t['last_persisted_completed_optimizer_steps'] == 200
