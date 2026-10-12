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


def test_second_seed_timeless_CE_is_matched_fixed_endpoint_not_a_dev_selected_variant():
    out = OUT.with_name('R_GT_window_v1_timeless_CE_s1028')
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '314b5f12811696184f6524f6ccf1466f9e4e6f94f568ba3df6f2f3e01cb3c5b5'
    r = json.loads((out / 'result.json').read_text()); paired = json.loads((OUT / 'result.json').read_text())
    assert r['seed'] == paired['seed'] == 1028 and r['variant'] == 'timeless_CE'
    assert r['completed_optimizer_steps'] == 200 and r['endpoint']['reopen_exact']
    assert r['model_parameter_sha256_before'] == paired['model_parameter_sha256_before']
    assert [(x['support_members'], x['support_observations']) for x in r['loss_curve']] == [
        (x['support_members'], x['support_observations']) for x in paired['loss_curve']]
    assert all(x['loss'] == x['Known_CE'] for x in r['loss_curve'])
    assert not r['dev_final_Val_Test_targets_or_pixels_used']
    terminal = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None


def test_second_seed_temporal_teacher_fixed_endpoint_and_matched_sampling():
    out = OUT.with_name('R_GT_window_v1_temporal_Raw_teacher_s1028')
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == 'e2df96a62d717cf35db4362d2ff6aa38acc11dace182ddddb1339aa94178791d'
    r = json.loads((out / 'result.json').read_text()); paired = json.loads((OUT / 'result.json').read_text())
    assert r['seed'] == 1028 and r['variant'] == 'temporal_Raw_teacher'
    assert r['completed_optimizer_steps'] == 200 and r['endpoint']['reopen_exact']
    assert r['model_parameter_sha256_before'] == paired['model_parameter_sha256_before']
    assert [(x['support_members'], x['support_observations']) for x in r['loss_curve']] == [
        (x['support_members'], x['support_observations']) for x in paired['loss_curve']]
    assert not r['dev_final_Val_Test_targets_or_pixels_used']
    terminal = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None


def test_second_seed_all_four_registered_R_variants_completed_and_paired_sampling():
    out = OUT.with_name('R_GT_window_v1_timeless_Raw_teacher_s1028')
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == 'd055c0f41a4b930ae03d5f9c340e10a074908a86c46d70ff7255d20d3705770c'
    r = json.loads((out / 'result.json').read_text())
    for variant in ('temporal_CE', 'timeless_CE', 'temporal_Raw_teacher', 'timeless_Raw_teacher'):
        other_out = OUT.with_name('R_GT_window_v1_' + variant + '_s1028')
        other = json.loads((other_out / 'result.json').read_text())
        assert other['completed_optimizer_steps'] == 200 and other['seed'] == 1028
        assert other['model_parameter_sha256_before'] == r['model_parameter_sha256_before']
        assert [(x['support_members'], x['support_observations']) for x in other['loss_curve']] == [
            (x['support_members'], x['support_observations']) for x in r['loss_curve']]
        assert other['endpoint']['reopen_exact'] and not other['dev_final_Val_Test_targets_or_pixels_used']
        t = json.loads(other_out.with_name(other_out.name + '_supervisor_attempt_1.json').read_text())
        assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
