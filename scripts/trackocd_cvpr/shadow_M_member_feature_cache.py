#!/usr/bin/env python3
"""Actual full-policy ALL11 old/new input memo parity, not full-dev retry."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import shadow_M_raw_cache as runtime
from src.trackocd_cvpr.M_all_model_fingerprint_development_profile import verify_all_model_shadow, SOURCES as OLD_SOURCES
from src.trackocd_cvpr.frozen_member_feature_cache_scope import frozen_member_feature_cache_scope
from src.trackocd_v2.io import atomic_json, sha256_file

SCRIPT = 'scripts/trackocd_cvpr/shadow_M_member_feature_cache.py'
CONFIG = 'configs/trackocd_cvpr/M_member_feature_cache_shadow_v1.json'
ORIGINAL_CONFIG = runtime.CONFIG
PRIOR = 'outputs/trackocd_cvpr/cvpr4/M_all_model_fingerprint_shadow_v1/result.json'
FAILED = 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_all_model_fingerprints_v1_relative_quarantine_s1027_RAM_retry1_supervisor_attempt_1.json'
PROFILE = 'RAW32MiB_SHA15.5MiB_MEMBER_INPUT16MiB_TEMP_INDEX0.5MiB_TOTAL64MiB'
SOURCES = tuple(dict.fromkeys((*runtime.SOURCES, *OLD_SOURCES, SCRIPT, CONFIG, FAILED,
    'src/trackocd_cvpr/frozen_member_feature_cache_scope.py')))


def configuration_gate(root, cfg):
    base = json.loads((root / ORIGINAL_CONFIG).read_text())
    projected = {k: v for k, v in cfg.items() if k not in (
        'cache_execution_profile', 'qualified_all11_result_sha256',
        'preserved_full_M_RAM_retry_terminal_sha256', 'learned_output_memo_controller_refresh_skip_or_full_development_use')}
    if projected.pop('output_directory') != 'outputs/trackocd_cvpr/cvpr4/M_member_feature_cache_shadow_v1':
        raise ValueError('New immutable entire-policy shadow namespace')
    base.pop('output_directory')
    if (projected != base or cfg['cache_execution_profile'] != PROFILE
            or cfg['qualified_all11_result_sha256'] != '9b552ebd1f3919d2f697df034aa3843d4a98fa39b08ea8be53716a52b52f595f'
            or cfg['preserved_full_M_RAM_retry_terminal_sha256'] != '97b6ee70217340a4a4dd2a094469343ef6a74f03d25995f81bf09c9abf3193da'
            or cfg['learned_output_memo_controller_refresh_skip_or_full_development_use']):
        raise ValueError('All11 unchanged policy/models/clock/guards, input-only cache qualification')
    previous = verify_all_model_shadow(root)
    for path, expected in ((PRIOR, cfg['qualified_all11_result_sha256']),
            (FAILED, cfg['preserved_full_M_RAM_retry_terminal_sha256'])):
        if (root / path).is_symlink() or sha256_file(root / path) != expected:
            raise ValueError('Exact previous full-policy parity and preserved incomplete full-dev failure')
    terminal = json.loads((root / FAILED).read_text())
    if (terminal['successful_owned_execution'] or terminal['complete_result_present']
            or terminal['owned_worker_exit_code'] != -9 or terminal['guard_stop_reason'] != 'owned_job_wall'):
        raise ValueError('No partial old result salvage or failed namespace reuse')
    return previous


def main():
    from src.trackocd_cvpr import qualified_raw_cache_scope as original_scope
    original_gate, original_worker = runtime.source_gate, runtime.worker
    old_scope = original_scope.qualified_raw_cache_scope
    runtime.SCRIPT, runtime.CONFIG, runtime.SOURCES = SCRIPT, CONFIG, SOURCES
    records = []

    def gate(args, cfg, safety):
        configuration_gate(ROOT, cfg)
        return original_gate(args, cfg, safety)  # Fresh exact live/source/authority before private inputs.

    @contextmanager
    def scope():
        with frozen_member_feature_cache_scope() as (raw, sha, features, index):
            class Usage:
                @property
                def retained_bytes(self):
                    return raw.retained_bytes + sha.retained_bytes + features.retained_bytes + index.retained_bytes
            yield Usage()
            if index.snapshot is not None or index.retained_bytes:
                raise ValueError('Current snapshot references must be released after each refresh')
            records.append({'Raw_statistics_bytes': raw.retained_bytes, 'fingerprint_bytes': sha.retained_bytes,
                'fingerprint_hits': sha.hits, 'fingerprint_misses': sha.misses,
                'member_input_bytes': features.retained_bytes, 'member_input_hits': features.hits,
                'member_input_misses': features.misses, 'member_input_fallbacks': features.fallbacks,
                'temporary_index_peak_bookkeeping_bytes': index.peak_bytes,
                'temporary_index_hits': index.hits, 'temporary_index_validated_snapshots': index.misses,
                'temporary_index_fallbacks': index.fallbacks})

    def worker(args, cfg, safety):
        original_worker(args, cfg, safety)  # All11 complete OLD and NEW actual fresh replays.
        out = ROOT / cfg['output_directory']
        result = json.loads((out / 'result.json').read_text())
        previous = json.loads((ROOT / PRIOR).read_text())
        if (len(records) != 11 or set(result['cases']) != set(previous['cases'])
                or any(result['cases'][key]['private_trace_sha256'] != previous['cases'][key]['private_trace_sha256']
                    for key in previous['cases'])):
            raise ValueError('ALL11 whole event/score/version/ledger traces also match previous real qualification')
        names = ['Mean_latest', 'StateEMA_latest'] + [v + '_s' + str(s) for v in runtime.VARIANTS for s in runtime.SEEDS]
        for name, counters in zip(names, records):
            counters['case'] = name
            if (counters['Raw_statistics_bytes'] > 33554432 or counters['fingerprint_bytes'] > 16252928
                    or counters['member_input_bytes'] > 16777216
                    or counters['temporary_index_peak_bookkeeping_bytes'] > 524288
                    or counters['temporary_index_hits'] < 1 or counters['fingerprint_hits'] < 1
                    or (name not in ('Mean_latest', 'StateEMA_latest') and counters['member_input_hits'] < 1)):
                raise ValueError('Each actual cache scope exercised within the unchanged64MiB logical ceiling')
        atomic_json(out / 'execution_profile.json', {'profile': PROFILE,
            'result_sha256': sha256_file(out / 'result.json'), 'ordered_all11_scope_counters': records,
            'actual_all11_trace_digests_match_previous_profile': True,
            'learned_logits_prototypes_predictions_or_model_outputs_cached': False,
            'controller_refreshes_history_members_probes_or_numeric_kernels_changed': False,
            'raw_cache_python_allocator_bytes_equal_logical_ceiling_claim': False,
            'full_development_parity_or_speedup_guarantee_or_Goal_accepted': False})

    runtime.source_gate, runtime.worker = gate, worker
    original_scope.qualified_raw_cache_scope = scope
    try:
        return runtime.main()
    finally:
        original_scope.qualified_raw_cache_scope = old_scope


if __name__ == '__main__':
    sys.exit(main())
