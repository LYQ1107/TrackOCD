"""Actual fixed endpoints, not a scientific acceptance or synthetic fit claim."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HASHES = {
    'temporal_Raw_teacher': '981fc224e844d4531a7e97a3fd9db54a9967b76550a1c93a82efd69c701dfcee',
    'timeless_Raw_teacher': '878b8adc1cd81d8022a77323da28cda1e3f7d37d9cfba382fb03b5dedb1afbe9',
    'temporal_CE': '389da9973dbca675352e71e96b816bc7ca946f34672a51ea018d444095295a75',
    'timeless_CE': '995fe30f1f3eba279c29f7be7e341435239b4c670f7986b22b370801fa924606',
}


def test_four_actual_fixed_endpoints_share_seed_schedule_and_fit_scope():
    reference = None
    for variant, expected in HASHES.items():
        out = ROOT / ('outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_' + variant + '_s1027')
        path = out / 'result.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        r = json.loads(path.read_text())
        terminal = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
        assert r['variant'] == variant and r['seed'] == 1027
        assert r['completed_optimizer_steps'] == 200
        assert r['checkpoint_selection'] == 'Fixed200 endpoint' and r['endpoint']['reopen_exact']
        assert r['label_and_physical_scope']['physical_tracks_kept'] == 1401
        assert r['label_and_physical_scope']['eligible_supervised_tracks'] == 1158
        assert r['label_and_physical_scope']['masked_no_fit5_target_tracks'] == 243
        assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
        assert terminal['guard_stop_reason'] is None
        assert not r['resources']['CUDA_initialized'] and r['resources']['training_GPU_hours'] == 0
        assert not r['dev_final_Val_Test_targets_or_pixels_used']
        assert not r['all3seeds_or_distinct_R_C_M_mechanisms_or_Goal_A_B_C_accepted']
        footprint = [(row['support_members'], row['support_observations']) for row in r['loss_curve']]
        assert [row['step'] for row in r['loss_curve']] == list(range(1, 201))
        if reference is None:
            reference = (r['model_parameter_sha256_before'], footprint)
        assert (r['model_parameter_sha256_before'], footprint) == reference
        if variant.endswith('_CE'):
            assert all(row['loss'] == row['Known_CE'] for row in r['loss_curve'])
