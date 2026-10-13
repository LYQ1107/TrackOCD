"""Actual incomplete full-M attempt is preserved, never metric acceptance."""
import json
from pathlib import Path
from src.trackocd_v2.io import sha256_file

ROOT = Path(__file__).resolve().parents[2]
NS = 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_member_input_memo_v1_relative_quarantine_s1027'


def test_actual_input_memo_full_M_wall_stop_no_result_or_profile():
    out = ROOT / NS
    terminal = out.with_name(out.name + '_supervisor_attempt_1.json')
    assert sha256_file(terminal) == '5fa18163351e49df16b5e04102ed82500c2396cb7618b62d68a5122ffb9c359f'
    assert sha256_file(out / 'progress.json') == '8374eb69acac2d25ba02027bd3de12cd556a12ab32d71af8448229d41c295ed9'
    result = json.loads(terminal.read_text())
    assert result['owned_worker_exit_code'] == -9 and result['guard_stop_reason'] == 'owned_job_wall'
    assert not result['successful_owned_execution'] and not result['complete_result_present']
    assert result['only_new_owned_Popen_worker_managed'] and result['training_GPU_hours'] == 0
    assert 3000 <= result['wall_seconds_including_imports_and_network'] < 3002
    assert result['peak_sampled_owned_worker_RSS_bytes'] < 3 * 1024**3
    assert result['peak_sampled_worker_threads'] == 1
    assert {p.name for p in out.iterdir()} == {'launch.json', 'progress.json', 'owned_startup_observation_v1.json'}
    launch = json.loads((out / 'launch.json').read_text())
    assert launch['live_remote_registration_proof']['sha'] == 'c099c5ac6e7f575b55f6f9a2f84284cbab26990e'
    assert json.loads((out / 'progress.json').read_text()) == {
        'completed_order': 1029, 'no_optimizer': True, 'physical_tracks': 472}
    # Equal progress framing is not whole-trace equality or completion of probes.
    assert not (out / 'result.json').exists() and not (out / 'execution_profile.json').exists()
