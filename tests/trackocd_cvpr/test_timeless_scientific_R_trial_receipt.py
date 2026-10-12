import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_timeless_Raw_teacher_s1027'


def test_matched_timeless_trial_endpoint_and_schedule_not_selected_by_loss():
    p = OUT / 'result.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == '878b8adc1cd81d8022a77323da28cda1e3f7d37d9cfba382fb03b5dedb1afbe9'
    r = json.loads(p.read_text())
    t = json.loads((ROOT / 'outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_temporal_Raw_teacher_s1027/result.json').read_text())
    assert r['variant'] == 'timeless_Raw_teacher' and r['seed'] == t['seed'] == 1027
    assert r['completed_optimizer_steps'] == t['completed_optimizer_steps'] == 200
    assert r['model_parameter_sha256_before'] == t['model_parameter_sha256_before']
    assert r['endpoint']['reopen_exact'] and r['checkpoint_selection'] == 'Fixed200 endpoint'
    assert [(x['support_members'], x['support_observations']) for x in r['loss_curve']] == [
        (x['support_members'], x['support_observations']) for x in t['loss_curve']]
    assert not r['dev_final_Val_Test_targets_or_pixels_used']
    assert not r['all3seeds_or_distinct_R_C_M_mechanisms_or_Goal_A_B_C_accepted']
    terminal = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
