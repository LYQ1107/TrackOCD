"""Actual first formal fit: public receipts only, not development acceptance."""
import json
import math
from pathlib import Path
from src.trackocd_v2.io import sha256_file

ROOT = Path(__file__).resolve().parents[2]
NS = 'outputs/trackocd_cvpr/cvpr2/fresh_A_fit5_Fixed200_v1_A1_ADAPTER_s1027'


def test_actual_first_fresh_fixed200_complete_not_all15_or_dev():
    out = ROOT / NS
    hashes = {
        out / 'result.json': '089dbeb6cdb0360de9959c22a484e1347d818c5705485e6fe40f296404106845',
        out / 'launch.json': '2e52f3434d08a91efa3ac76c34f41ae9b4638a1a0447a5e3a07cb8e7f81a7bfc',
        out.with_name(out.name + '_supervisor_attempt_1.json'): '633bd965a9adedb119350ba83982d12e7c95cc1c4d127bf216dab0070dc4baf9',
    }
    for path, expected in hashes.items():
        assert sha256_file(path) == expected
    result, launch, terminal = (json.loads(path.read_text()) for path in hashes)
    trial = result['trial']
    assert launch['preregistration_commit'] == '2a51b62feb5273bd7541773d0afcd8e04e00ff79'
    assert launch['live_remote_registration_proof'] == result['live_remote_registration_proof']
    assert result['live_remote_registration_proof']['sha'] == launch['preregistration_commit']
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None and terminal['full_result_present']
    assert terminal['last_persisted_completed_optimizer_steps'] == 200
    assert terminal['peak_sampled_owned_worker_RSS_bytes'] < 3 * 1024**3
    assert terminal['wall_seconds_including_worker_imports_and_network'] < 3000
    assert terminal['CPU_compute_workers'] == 1 and terminal['peak_sampled_worker_threads'] <= 2
    assert terminal['peak_identified_native_driver_helpers'] <= 1
    assert terminal['training_GPU_hours'] == result['resources']['training_GPU_hours'] == 0
    assert not result['resources']['CUDA_initialized']
    assert trial['family'] == 'A1_ADAPTER' and trial['seed'] == 1027 and trial['stage'] == 'Fixed200'
    assert trial['completed_optimizer_steps'] == 200 and trial['parameter_count'] == 525056
    assert trial['model_parameter_sha256_before'] != trial['model_parameter_sha256_after']
    assert trial['adapter_loss_gradient_norm_peak'] > 0
    assert [r['step'] for r in trial['loss_curve']] == list(range(1, 201))
    assert all(math.isfinite(r['loss']) and r['adapter_loss_gradient_norm'] > 0 for r in trial['loss_curve'])
    assert trial['scope']['physical_tracks'] == 1401
    assert trial['scope']['eligible_known_physical_tracks'] == 1158
    assert trial['scope']['masked_no_fit5_target_tracks'] == 243
    assert trial['actually_sampled_canonical_groups'] == 498 and trial['actually_sampled_videos'] == 226
    assert [c['step'] for c in result['private_checkpoints']] == [20, 50, 100, 200]
    assert all(c['reopen_exact'] for c in result['private_checkpoints'])
    assert result['checkpoint_selection'] == 'Fixed200, never dev/final/Val/Test selection'
    assert not result['pilot_or_historical_checkpoint_warmstart']
    assert not result['dev_final_Val_Test_inputs_used'] and not result['formal15fits_dev_or_Goal_accepted']
    # No private weight reopening or independent full-development claim here.
