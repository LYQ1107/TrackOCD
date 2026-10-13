#!/usr/bin/env python3
"""NEW actual ALL11 exact Raw fingerprint shadow, including every trained M.

Original persistent supervisor/worker does ALL full OLD/cache replays and exact
trace/outcome comparisons. Fresh source/data/authority/cohort gates, original
numeric models/history/protocol/limits. No full dev use before actual parity.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import shadow_M_raw_cache as runtime
from src.trackocd_cvpr.M_fingerprint_development_profile import verify_fingerprint_shadow, SOURCES as CACHE_SOURCES
from src.trackocd_cvpr.qualified_M_fingerprint_cache_scope import qualified_M_fingerprint_cache_scope
from src.trackocd_v2.io import atomic_json, sha256_file

SCRIPT = 'scripts/trackocd_cvpr/shadow_M_all_model_fingerprints.py'
CONFIG = 'configs/trackocd_cvpr/M_all_model_fingerprint_shadow_v1.json'
BASE_CONFIG = runtime.CONFIG
PROFILE = 'outputs/trackocd_cvpr/cvpr4/M_policy_cpu_profile_v1'
PROFILE_SHA = 'bb821f9120bd45b64ed2e52cc4b415edf10445b07dcefa9f187f0c59e868cada'
OLD_SHADOW = 'outputs/trackocd_cvpr/cvpr4/M_fingerprint_cache_controller_shadow_v1/result.json'
CACHE_PROFILE = 'EXACT_RAW_SUPPORT_SHA_ALL11_32MiB_AND_FROZEN_RAW_32MiB_TOTAL_64MiB'
SOURCES = (SCRIPT, CONFIG, 'src/trackocd_cvpr/qualified_M_fingerprint_cache_scope.py',
    PROFILE + '/launch.json', PROFILE + '/result.json', PROFILE + '_supervisor_attempt_1.json',
    'scripts/trackocd_cvpr/profile_M_policy.py', 'configs/trackocd_cvpr/M_policy_cpu_profile_v1.json',
    'src/trackocd_cvpr/public_profile_summary.py', *CACHE_SOURCES)


def verify_profile_configuration(root, cfg):
    base = json.loads((root / BASE_CONFIG).read_text())
    projected = {k: v for k, v in cfg.items() if k not in ('cache_execution_profile', 'profile_evidence_sha256')}
    if projected.pop('output_directory') != 'outputs/trackocd_cvpr/cvpr4/M_all_model_fingerprint_shadow_v1':
        raise ValueError('New registered ALL11 fingerprint scope, never old namespace')
    base.pop('output_directory')
    if projected != base or cfg['cache_execution_profile'] != CACHE_PROFILE or cfg['profile_evidence_sha256'] != PROFILE_SHA:
        raise ValueError('Entire original ALL11 protocol/data/models/guards required')
    qualified = verify_fingerprint_shadow(root)
    if (root / (PROFILE + '/result.json')).is_symlink() or sha256_file(root / (PROFILE + '/result.json')) != PROFILE_SHA:
        raise ValueError('Actual complete frozen-M policy profiling required')
    t = json.loads((root / (PROFILE + '_supervisor_attempt_1.json')).read_text())
    r = json.loads((root / (PROFILE + '/result.json')).read_text())
    launch = json.loads((root / (PROFILE + '/launch.json')).read_text())
    if (not t['successful_owned_execution'] or t['owned_worker_exit_code'] != 0 or t['guard_stop_reason'] is not None
            or not t['complete_result_present'] or t['wall_seconds_including_registration'] > 3000
            or t['peak_sampled_owned_worker_RSS_bytes'] > 3221225472
            or r['live_remote_registration_proof']['sha'] != 'b40a446f608dd48e0d1c6634262ed285d628290a'
            or not r['actual_complete_trace_exact_previous_shadow'] or not r['model_state_unchanged']):
        raise ValueError('Exact normally completed profile, not partial timing or selected endpoint')
    for p in ('scripts/trackocd_cvpr/profile_M_policy.py', 'configs/trackocd_cvpr/M_policy_cpu_profile_v1.json',
              'src/trackocd_cvpr/qualified_fingerprint_cache_scope.py', 'src/trackocd_cvpr/member_reliability.py',
              'src/trackocd_cvpr/versioned_reliability_memory.py'):
        if launch['source_sha256'][p] != sha256_file(root / p):
            raise ValueError('Exact original profiled kernel/scopes required')
    return qualified


def main():
    from src.trackocd_cvpr import qualified_raw_cache_scope as original_scope
    original_gate, original_worker = runtime.source_gate, runtime.worker
    old_scope = original_scope.qualified_raw_cache_scope
    runtime.SCRIPT, runtime.CONFIG = SCRIPT, CONFIG
    runtime.SOURCES = tuple(dict.fromkeys((*runtime.SOURCES, *SOURCES)))
    records = []
    def gate(args, cfg, safety):
        verify_profile_configuration(ROOT, cfg)
        return original_gate(args, cfg, safety)
    @contextmanager
    def scope():
        with qualified_M_fingerprint_cache_scope() as (raw, fingerprints):
            class Usage:
                @property
                def retained_bytes(self): return raw.retained_bytes + fingerprints.retained_bytes
            yield Usage()
            records.append({'Raw_statistics_bytes': raw.retained_bytes,
                'fingerprint_bytes': fingerprints.retained_bytes,
                'fingerprint_hits': fingerprints.hits, 'fingerprint_misses': fingerprints.misses})
    def worker(args, cfg, safety):
        original_worker(args, cfg, safety)
        if len(records) != 11 or any(row['Raw_statistics_bytes'] > 33554432
                or row['fingerprint_bytes'] > 33554432 or not row['fingerprint_hits'] for row in records):
            raise ValueError('Actual ALL11 hashing scopes exercised within unchanged shared64MiB')
        out = ROOT / cfg['output_directory']; result = json.loads((out / 'result.json').read_text())
        prior = json.loads((ROOT / OLD_SHADOW).read_text())
        if set(result['cases']) != set(prior['cases']) or any(
                result['cases'][name]['private_trace_sha256'] != prior['cases'][name]['private_trace_sha256']
                for name in prior['cases']):
            raise ValueError('ALL11 whole traces must ALSO match previous qualified profile')
        atomic_json(out / 'execution_profile.json', {'profile': CACHE_PROFILE,
            'result_sha256': sha256_file(out / 'result.json'), 'ordered_all11_scope_counters': records,
            'actual_all11_trace_digests_match_previous_profile': True,
            'global_hashlib_numeric_hashes_models_data_thresholds_orders_history_and_guards_unchanged': True,
            'full72_development_speedup_purity_or_Goal_accepted': False})
    runtime.source_gate, runtime.worker = gate, worker
    original_scope.qualified_raw_cache_scope = scope
    try: return runtime.main()
    finally: original_scope.qualified_raw_cache_scope = old_scope


if __name__ == '__main__': sys.exit(main())
