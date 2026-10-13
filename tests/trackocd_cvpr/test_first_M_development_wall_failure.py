import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_actual_first_trained_M_wall_failure_stays_incomplete_not_two_order_science():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_fingerprint_cache_v1_relative_quarantine_s1027'
    terminal = out.with_name(out.name + '_supervisor_attempt_1.json')
    assert hashlib.sha256(terminal.read_bytes()).hexdigest() == '3c4592996c88992fac9cd02c28fe2df38d962b667b735492901d5a45d9ad643e'
    t = json.loads(terminal.read_text())
    assert not t['successful_owned_execution'] and not t['complete_result_present']
    assert t['owned_worker_exit_code'] == -9 and t['guard_stop_reason'] == 'owned_job_wall'
    assert t['only_new_owned_Popen_worker_managed'] and t['training_GPU_hours'] == 0
    assert t['wall_seconds_including_imports_and_network'] >= 3000
    assert t['peak_sampled_worker_threads'] == 1 and t['peak_sampled_owned_worker_RSS_bytes'] < 3221225472
    assert not (out / 'result.json').exists() and not (out / 'execution_profile.json').exists()
    assert json.loads((out / 'progress.json').read_text()) == {
        'completed_order': 1028, 'no_optimizer': True, 'physical_tracks': 472}
    launch = json.loads((out / 'launch.json').read_text())
    assert launch['live_remote_registration_proof']['sha'] == '5b487b663059a6d5d2a3869aa175f1df8c6b8825'
