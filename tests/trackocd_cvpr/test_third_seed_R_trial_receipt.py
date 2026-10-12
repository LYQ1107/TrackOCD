import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_temporal_CE_s1029'


def test_third_R_seed_is_actual_fresh_fixed_endpoint_not_reused_dev_winner():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '5f21e0f1147ede151fb3ccf19566c07660447ab1d46bea936f6c401c648ca506'
    r = json.loads((OUT / 'result.json').read_text())
    assert r['seed'] == 1029 and r['variant'] == 'temporal_CE'
    assert r['completed_optimizer_steps'] == 200 and r['checkpoint_selection'] == 'Fixed200 endpoint'
    assert r['endpoint'] == {'bytes': 616115, 'sha256': 'b31db960e730c52994646f026bdc49897111bdd82d86b9afec211c60a599ceff', 'reopen_exact': True}
    for seed in (1027, 1028):
        old = json.loads((OUT.with_name('R_GT_window_v1_temporal_CE_s' + str(seed)) / 'result.json').read_text())
        assert old['model_parameter_sha256_before'] != r['model_parameter_sha256_before']
        assert old['model_parameter_sha256_after'] != r['model_parameter_sha256_after']
    assert len(r['loss_curve']) == 200 and all(x['loss'] == x['Known_CE'] for x in r['loss_curve'])
    assert r['label_and_physical_scope']['physical_tracks_kept'] == 1401
    assert r['label_and_physical_scope']['eligible_supervised_tracks'] == 1158
    assert r['label_and_physical_scope']['masked_no_fit5_target_tracks'] == 243
    assert not r['old_fitted_adapter_initialization'] and not r['dev_final_Val_Test_targets_or_pixels_used']
    assert not r['all3seeds_or_distinct_R_C_M_mechanisms_or_Goal_A_B_C_accepted']
    assert not r['resources']['CUDA_initialized'] and r['resources']['training_GPU_hours'] == 0
    terminal = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None and terminal['last_persisted_completed_optimizer_steps'] == 200


def test_third_seed_static_CE_fixed_endpoint_and_sampling_match_temporal():
    out = OUT.with_name('R_GT_window_v1_timeless_CE_s1029')
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '5728502a03ec131ea9b7db41d6a224364eaa2573fea68790ca96dc61714c0e24'
    r = json.loads((out / 'result.json').read_text()); pair = json.loads((OUT / 'result.json').read_text())
    assert r['seed'] == pair['seed'] == 1029 and r['variant'] == 'timeless_CE'
    assert r['completed_optimizer_steps'] == 200 and r['checkpoint_selection'] == 'Fixed200 endpoint'
    assert r['model_parameter_sha256_before'] == pair['model_parameter_sha256_before']
    assert [(x['support_members'], x['support_observations']) for x in r['loss_curve']] == [
        (x['support_members'], x['support_observations']) for x in pair['loss_curve']]
    assert all(x['loss'] == x['Known_CE'] for x in r['loss_curve'])
    assert r['endpoint'] == {'bytes': 616115, 'sha256': 'd4046cdf07a885b01f5968787ccd2eba53633fc2e4757cd98e3ea5a1f8feb405', 'reopen_exact': True}
    terminal = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None and terminal['last_persisted_completed_optimizer_steps'] == 200


def test_third_seed_temporal_teacher_fresh_fixed_endpoint_and_sampling():
    out = OUT.with_name('R_GT_window_v1_temporal_Raw_teacher_s1029')
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '88c26b6aa46e49bc8d63f7f8a9bef16372995fb4563b1704710087a13d9bf2ac'
    r = json.loads((out / 'result.json').read_text()); pair = json.loads((OUT / 'result.json').read_text())
    assert r['seed'] == pair['seed'] == 1029 and r['variant'] == 'temporal_Raw_teacher'
    assert r['completed_optimizer_steps'] == 200 and r['checkpoint_selection'] == 'Fixed200 endpoint'
    assert r['model_parameter_sha256_before'] == pair['model_parameter_sha256_before']
    assert [(x['support_members'], x['support_observations']) for x in r['loss_curve']] == [
        (x['support_members'], x['support_observations']) for x in pair['loss_curve']]
    assert r['endpoint'] == {'bytes': 616115, 'sha256': '1f3bacdabdc901db7d8e900d0761ad46937d8840f2bc595d89bc09d18c658b2f', 'reopen_exact': True}
    assert not r['dev_final_Val_Test_targets_or_pixels_used'] and not r['old_fitted_adapter_initialization']
    terminal = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None and terminal['last_persisted_completed_optimizer_steps'] == 200
