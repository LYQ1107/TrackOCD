import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_actual_full_policy_profile_is_normal_exact_trace_not_full_development_or_speedup():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/Mean_policy_cpu_profile_v1'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '05bc1c47a49df515ab7639e68b2258d477eb9406cb811787e6f7ca27c162931c'
    r = json.loads((out / 'result.json').read_text())
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
    assert (r['physical_tracks'], r['observations'], r['metadata_frames'], r['empty_frames']) == (54, 1268, 374, 1)
    assert r['actual_complete_trace_exact_previous_shadow']
    assert r['private_trace_sha256'] == '7461015c0f64bf89c294630f08fac2cc8c8083a664b541c221d49bd5121f5ed5'
    assert not r['full_dev_bottleneck_isolated_speedup_or_scientific_Goal_accepted']
    assert not r['profile']['raw_profiler_callers_arguments_locals_or_private_strings_exported']
    assert r['optimizer_steps'] == 0 and r['resources']['training_GPU_hours'] == 0
    assert r['live_remote_registration_proof']['sha'] == 'bd2977ea621b8c775bf914dd4ece3f9bab5f3ec9'
    rows = r['profile']['top_registered_functions_by_overlapping_cumulative_time']
    assert any(row['source'].endswith('frame_atomic_state_ema.py') and row['function'] == 'refresh'
               and row['total_calls'] == 748 for row in rows)
