#!/usr/bin/env python3
"""NEW full ALL11 paired shadow of exact SHA memo + 32MiB Raw cache.

Original guarded driver performs every old/new whole-policy event/score/
version/ledger/outcome comparison; same scopes/limits, new namespace/profile.
Only own-process context binding differs; no original source modification.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import shadow_M_raw_cache as runtime
from src.trackocd_cvpr.qualified_fingerprint_cache_scope import qualified_fingerprint_cache_scope
from src.trackocd_v2.io import atomic_json, sha256_file

SCRIPT = 'scripts/trackocd_cvpr/shadow_M_fingerprint_cache.py'
CONFIG = 'configs/trackocd_cvpr/M_fingerprint_cache_controller_shadow_v1.json'
BASE_CONFIG = runtime.CONFIG
PROFILE = 'outputs/trackocd_cvpr/cvpr4/Mean_policy_cpu_profile_v1'
SOURCES = (SCRIPT, CONFIG, 'src/trackocd_cvpr/qualified_fingerprint_cache_scope.py',
    'configs/trackocd_cvpr/source_fingerprint_cache_scope_v1.json',
    PROFILE + '/launch.json', PROFILE + '/result.json', PROFILE + '_supervisor_attempt_1.json')


def verify_profile_configuration(root, cfg):
    base = json.loads((root / BASE_CONFIG).read_text())
    extras = {'cache_execution_profile', 'profile_evidence', 'profile_evidence_sha256'}
    projected = {key: value for key, value in cfg.items() if key not in extras}
    if projected.pop('output_directory') != 'outputs/trackocd_cvpr/cvpr4/M_fingerprint_cache_controller_shadow_v1':
        raise ValueError('New fixed fingerprint shadow namespace required')
    base.pop('output_directory')
    if projected != base or cfg['cache_execution_profile'] != 'EXACT_BYTE_SEGMENT_SHA_MEMO_32MiB_AND_FROZEN_RAW_32MiB_TOTAL_64MiB':
        raise ValueError('Same entire ALL11 protocol/guards, only explicit derived cache profile differs')
    if (cfg['profile_evidence'] != PROFILE + '/result.json'
            or cfg['profile_evidence_sha256'] != '05bc1c47a49df515ab7639e68b2258d477eb9406cb811787e6f7ca27c162931c'
            or sha256_file(root / cfg['profile_evidence']) != cfg['profile_evidence_sha256']):
        raise ValueError('Exact actual policy profile required')
    t = json.loads((root / (PROFILE + '_supervisor_attempt_1.json')).read_text())
    if not t['successful_owned_execution'] or t['owned_worker_exit_code'] != 0 or t['guard_stop_reason'] is not None:
        raise ValueError('Normal actual completed profiling required')


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
        with qualified_fingerprint_cache_scope() as (raw, fingerprints):
            class Usage:
                @property
                def retained_bytes(self): return raw.retained_bytes + fingerprints.retained_bytes
            yield Usage()
            records.append({'Raw_statistics_bytes': raw.retained_bytes,
                'fingerprint_bytes': fingerprints.retained_bytes,
                'fingerprint_hits': fingerprints.hits, 'fingerprint_misses': fingerprints.misses})
    def worker(args, cfg, safety):
        original_worker(args, cfg, safety)
        if len(records) != 11 or any(row['Raw_statistics_bytes'] + row['fingerprint_bytes'] > 67108864 for row in records):
            raise ValueError('All11 actual profile scopes and unchanged combined derived-cache bound')
        out = ROOT / cfg['output_directory']
        atomic_json(out / 'execution_profile.json', {'profile': cfg['cache_execution_profile'],
            'result_sha256': sha256_file(out / 'result.json'), 'ordered_all11_scope_counters': records,
            'global_hashlib_original_sources_data_models_thresholds_orders_and_guards_unchanged': True,
            'full72_development_speedup_purity_or_Goal_accepted': False})
    runtime.source_gate, runtime.worker = gate, worker
    original_scope.qualified_raw_cache_scope = scope
    try: return runtime.main()
    finally: original_scope.qualified_raw_cache_scope = old_scope


if __name__ == '__main__': sys.exit(main())
