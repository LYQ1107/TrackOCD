"""Actual public terminal failure, not a partial scientific result."""
import json
from pathlib import Path
from src.trackocd_v2.io import sha256_file

ROOT=Path(__file__).resolve().parents[2]
NS='outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_all_model_fingerprints_v1_relative_quarantine_s1027_RAM_retry1'


def test_actual_full_M_recovery_three_main_orders_do_not_imply_probe_or_result_completion():
    p=ROOT/NS;t=p.with_name(p.name+'_supervisor_attempt_1.json')
    assert sha256_file(t)=='97b6ee70217340a4a4dd2a094469343ef6a74f03d25995f81bf09c9abf3193da'
    assert sha256_file(p/'progress.json')=='8374eb69acac2d25ba02027bd3de12cd556a12ab32d71af8448229d41c295ed9'
    terminal=json.loads(t.read_text());progress=json.loads((p/'progress.json').read_text())
    assert terminal['owned_worker_exit_code']==-9 and terminal['guard_stop_reason']=='owned_job_wall'
    assert not terminal['successful_owned_execution'] and not terminal['complete_result_present']
    assert terminal['only_new_owned_Popen_worker_managed'] and terminal['training_GPU_hours']==0
    assert terminal['peak_sampled_worker_threads']==1 and terminal['peak_sampled_owned_worker_RSS_bytes']<3221225472
    assert 3000<=terminal['wall_seconds_including_imports_and_network']<3002
    assert progress=={'completed_order':1029,'no_optimizer':True,'physical_tracks':472}
    assert {f.name for f in p.iterdir()}=={'launch.json','progress.json'}
    assert not (p/'result.json').exists() and not (p/'execution_profile.json').exists()
    assert json.loads((p/'launch.json').read_text())['live_remote_registration_proof']['sha']=='12114836cd8ede90899e99245f3a41f4a227fa85'
