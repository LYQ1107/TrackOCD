"""Validate every available actual public fit, without private weight reads.

Partial receipts are checked as partial engineering evidence, never promoted
to an all15 scientific matrix. Tests do not cause or infer missing executions.
"""
import json
import math
from pathlib import Path
from src.trackocd_v2.io import sha256_file
from src.trackocd_cvpr.fresh_A_baseline import FAMILIES
from src.trackocd_cvpr.fresh_A_formal_fit import PREFIX
from scripts.trackocd_cvpr.fit_fresh_A_fixed200 import SOURCES

ROOT = Path(__file__).resolve().parents[2]


def test_available_actual_fixed200_public_receipts():
    completed = 0
    initializations, calendars = {}, {}
    for family in FAMILIES:
        for seed in (1027, 1028, 1029):
            out = ROOT / f'{PREFIX}_{family}_s{seed}'
            terminal_path = out.with_name(out.name + '_supervisor_attempt_1.json')
            if not terminal_path.exists():
                continue
            terminal = json.loads(terminal_path.read_text())
            assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
            assert terminal['guard_stop_reason'] is None and terminal['full_result_present']
            assert terminal['last_persisted_completed_optimizer_steps'] == 200
            assert terminal['wall_seconds_including_worker_imports_and_network'] < 3000
            assert terminal['peak_sampled_owned_worker_RSS_bytes'] < 3 * 1024**3
            assert terminal['CPU_compute_workers'] == 1 and terminal['peak_sampled_worker_threads'] <= 2
            assert terminal['peak_identified_native_driver_helpers'] <= 1
            launch = json.loads((out / 'launch.json').read_text())
            result = json.loads((out / 'result.json').read_text())
            trial = result['trial']
            assert result['status'] == 'COMPLETED_ONE_FRESH_FIXED200_FIT_NOT_ALL15_DEV_OR_GOAL_ACCEPTANCE'
            assert trial['family'] == family and trial['seed'] == seed and trial['stage'] == 'Fixed200'
            assert trial['completed_optimizer_steps'] == 200
            assert trial['parameter_count'] == (541889 if family.startswith('A2_') else 525056)
            assert trial['scope']['physical_tracks'] == 1401
            assert trial['scope']['eligible_known_physical_tracks'] == 1158
            assert trial['scope']['masked_no_fit5_target_tracks'] == 243
            assert trial['scope']['canonical_known_groups'] == 1158
            assert trial['scope']['supervised_classes'] == 5
            assert trial['model_parameter_sha256_before'] != trial['model_parameter_sha256_after']
            assert trial['adapter_loss_gradient_norm_peak'] > 0
            assert [r['step'] for r in trial['loss_curve']] == list(range(1, 201))
            assert all(math.isfinite(r['loss']) and r['adapter_loss_gradient_norm'] > 0 for r in trial['loss_curve'])
            if family.startswith('A2_'):
                assert trial['extra_loss_gradient_norm_peak'] > 0
            assert [r['step'] for r in result['private_checkpoints']] == [20, 50, 100, 200]
            assert all(r['reopen_exact'] for r in result['private_checkpoints'])
            assert launch['preregistration_commit'] == result['live_remote_registration_proof']['sha']
            assert launch['live_remote_registration_proof'] == result['live_remote_registration_proof']
            assert set(launch['source_sha256']) == set(SOURCES)
            for source in SOURCES:
                assert sha256_file(ROOT / source) == launch['source_sha256'][source]
            for lookup, key in ((initializations, 'adapter_initialization_sha256'), (calendars, 'batch_calendar_sha256')):
                if seed in lookup:
                    assert lookup[seed] == trial[key]
                lookup[seed] = trial[key]
            assert not result['pilot_or_historical_checkpoint_warmstart']
            assert not result['dev_final_Val_Test_inputs_used'] and not result['formal15fits_dev_or_Goal_accepted']
            assert terminal['training_GPU_hours'] == result['resources']['training_GPU_hours'] == trial['training_GPU_hours'] == 0
            assert not result['resources']['CUDA_initialized']
            completed += 1
    assert 1 <= completed <= 15
    # A changing count does not supply the independent cohort/live/dev gates.
