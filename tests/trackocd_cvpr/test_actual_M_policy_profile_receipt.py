import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_actual_fixed_M_complete_profile_normal_owned_terminal_and_exact_private_trace():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_policy_cpu_profile_v1'
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == 'bb821f9120bd45b64ed2e52cc4b415edf10445b07dcefa9f187f0c59e868cada'
    r = json.loads((out / 'result.json').read_text())
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0
    assert t['guard_stop_reason'] is None and t['complete_result_present']
    assert t['wall_seconds_including_registration'] <= 3000
    assert t['peak_sampled_owned_worker_RSS_bytes'] <= 3221225472
    assert (r['physical_tracks'], r['observations'], r['metadata_frames'], r['empty_frames']) == (54, 1268, 374, 1)
    assert (r['model'], r['training_seed'], r['order_seed']) == ('relative_quarantine', 1027, 1027)
    assert r['actual_complete_trace_exact_previous_shadow'] and r['model_state_unchanged']
    assert r['private_trace_sha256'] == 'c98657973810452d4c4c00c8ad7bee98fa290c393a9e3de01f51a318c7a0ce68'
    assert r['preserved_failed_terminal_sha256'] == '3c4592996c88992fac9cd02c28fe2df38d962b667b735492901d5a45d9ad643e'
    assert r['fixed_endpoint_sha256'] == '6674a0fbefd25f9e1e5d12c00c9e5a9c7b6825abe43f14dc739be46983b68729'
    assert r['derived_cache_counters'] == {'Raw_statistics_bytes': 33549000,
        'fingerprint_bytes': 0, 'fingerprint_hits': 0, 'fingerprint_misses': 0}
    assert not r['full_dev_bottleneck_isolated_speedup_or_scientific_Goal_accepted']
    assert not r['profile']['raw_profiler_callers_arguments_locals_or_private_strings_exported']
    assert r['optimizer_steps'] == 0 and r['resources']['training_GPU_hours'] == 0
    assert not r['resources']['CUDA_initialized']
    assert r['live_remote_registration_proof']['sha'] == 'b40a446f608dd48e0d1c6634262ed285d628290a'
    rows = r['profile']['top_registered_functions_by_overlapping_cumulative_time']
    assert any(row['source'].endswith('versioned_reliability_memory.py') and row['function'] == 'refresh'
               and row['total_calls'] == 748 for row in rows)
    assert any(row['source'].endswith('member_reliability.py') and row['function'] == 'weighted_support'
               and row['total_calls'] == 16475 for row in rows)
