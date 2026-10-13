import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_actual_M_Mean_wall_failure_remains_incomplete_not_a_selected_two_order_result():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_v1_Mean_latest'
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert not t['successful_owned_execution'] and not t['complete_result_present']
    assert t['owned_worker_exit_code'] == -9 and t['guard_stop_reason'] == 'owned_job_wall'
    assert not (out / 'result.json').exists()
    assert json.loads((out / 'progress.json').read_text())['completed_order'] == 1028
    assert t['only_new_owned_Popen_worker_managed'] and t['training_GPU_hours'] == 0
    assert t['wall_seconds_including_imports_and_network'] >= 3000


def test_cached_Mean_wall_failure_preserves_all_three_order_progress_but_no_full_result():
    import hashlib
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_raw_cache_v1_Mean_latest'
    terminal = out.with_name(out.name + '_supervisor_attempt_1.json')
    assert hashlib.sha256(terminal.read_bytes()).hexdigest() == '04d07895d893c67b3fd6ef7a2b50ec0263f72156d3ab2f4cc88a9839819ab5e9'
    t = json.loads(terminal.read_text())
    assert not t['successful_owned_execution'] and not t['complete_result_present']
    assert t['owned_worker_exit_code'] == -9 and t['guard_stop_reason'] == 'owned_job_wall'
    assert not (out / 'result.json').exists() and not (out / 'execution_profile.json').exists()
    assert json.loads((out / 'progress.json').read_text()) == {
        'completed_order': 1029, 'no_optimizer': True, 'physical_tracks': 472}
    assert t['only_new_owned_Popen_worker_managed'] and t['training_GPU_hours'] == 0
    assert t['peak_sampled_worker_threads'] == 1
    assert t['peak_sampled_owned_worker_RSS_bytes'] <= 3221225472
    assert t['wall_seconds_including_imports_and_network'] >= 3000
    launch = json.loads((out / 'launch.json').read_text())
    assert launch['live_remote_registration_proof']['sha'] == '63aebf6b743694c84fd4a19a33025ad3c260ca5d'
