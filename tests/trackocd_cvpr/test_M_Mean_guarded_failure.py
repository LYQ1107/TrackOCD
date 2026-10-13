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
