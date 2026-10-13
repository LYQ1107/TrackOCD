import hashlib
import json
from pathlib import Path
from src.trackocd_cvpr.M_fingerprint_development_profile import verify_fingerprint_shadow

ROOT = Path(__file__).resolve().parents[2]


def test_actual_all11_fingerprint_shadow_full_trace_profile_and_normal_owned_completion():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_fingerprint_cache_controller_shadow_v1'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == 'd2626ad2d872c03e88e6ddbb40806aeb4bc3f0e0c312ed2f31f0a30cea09c552'
    assert hashlib.sha256((out / 'execution_profile.json').read_bytes()).hexdigest() == '54547695ba2668e93dd7cc1c09f3ff2fb8f0a83ff1d5020fe17ef747fb8267ab'
    checked = verify_fingerprint_shadow(ROOT)
    assert checked['actual_entire_ALL11_fingerprint_shadow_profile_checked']
    assert not checked['full_development_parity_speedup_or_partial_old_metric_reuse_claim']
    r = json.loads((out / 'result.json').read_text())
    e = json.loads((out / 'execution_profile.json').read_text())
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert len(r['cases']) == len(e['ordered_all11_scope_counters']) == 11
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0
    assert t['guard_stop_reason'] is None and t['complete_result_present']
    assert e['ordered_all11_scope_counters'][0]['fingerprint_hits'] == 15259
    assert e['ordered_all11_scope_counters'][1]['fingerprint_hits'] == 15972
    # Weighted-M hash alias intentionally remains original; don't falsely claim
    # that the new SHA memo accelerates these cases.
    assert all(row['fingerprint_hits'] == row['fingerprint_misses'] == row['fingerprint_bytes'] == 0
               for row in e['ordered_all11_scope_counters'][2:])
    assert r['optimizer_steps'] == 0 and r['resources']['training_GPU_hours'] == 0
