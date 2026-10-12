import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_temporal_CE_s1028'


def test_second_training_seed_is_actual_fixed_fit_not_reused_or_selected_endpoint():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '0a751b7aec4adc19bac75c9b8354eee4c0043f8b8e5d3834516ecfe2a15ac58b'
    r = json.loads((OUT / 'result.json').read_text())
    first = json.loads((OUT.with_name('R_GT_window_v1_temporal_CE_s1027') / 'result.json').read_text())
    assert r['seed'] == 1028 and r['variant'] == 'temporal_CE'
    assert r['completed_optimizer_steps'] == 200 and r['checkpoint_selection'] == 'Fixed200 endpoint'
    assert r['endpoint']['reopen_exact'] and r['endpoint']['bytes'] == 616115
    assert r['model_parameter_sha256_before'] != first['model_parameter_sha256_before']
    assert r['model_parameter_sha256_after'] != first['model_parameter_sha256_after']
    assert [(x['support_members'], x['support_observations']) for x in r['loss_curve']] != [
        (x['support_members'], x['support_observations']) for x in first['loss_curve']]
    assert all(x['loss'] == x['Known_CE'] for x in r['loss_curve'])
    assert not r['old_fitted_adapter_initialization'] and not r['dev_final_Val_Test_targets_or_pixels_used']
    assert not r['all3seeds_or_distinct_R_C_M_mechanisms_or_Goal_A_B_C_accepted']
    assert not r['resources']['CUDA_initialized']
    terminal = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None and terminal['last_persisted_completed_optimizer_steps'] == 200
