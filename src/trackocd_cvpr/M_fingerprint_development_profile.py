"""Actual ALL11 fingerprint-shadow public gate, fail closed while pending.

Source/config tests do not qualify a partial live shadow or create metrics.
Original Raw tensor/shadow prerequisites and both failed Mean receipts remain.
"""
import json
import math
import re
from src.trackocd_v2.io import sha256_file
from .M_cached_development_profile import verify_actual_shadow, SOURCES as RAW_SOURCES

CONFIG = 'configs/trackocd_cvpr/M_GT_causal_development_fingerprint_cache_v1.json'
SHADOW = 'outputs/trackocd_cvpr/cvpr4/M_fingerprint_cache_controller_shadow_v1'
OLD_SHADOW = 'outputs/trackocd_cvpr/cvpr4/M_raw_cache_controller_shadow_v1'
FAILED = 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_raw_cache_v1_Mean_latest_supervisor_attempt_1.json'
PROFILE = 'EXACT_BYTE_SEGMENT_SHA_MEMO_32MiB_AND_FROZEN_RAW_32MiB_TOTAL_64MiB'
SOURCES = tuple(dict.fromkeys((*RAW_SOURCES, CONFIG,
    'src/trackocd_cvpr/M_fingerprint_development_profile.py',
    'src/trackocd_cvpr/qualified_fingerprint_cache_scope.py',
    'configs/trackocd_cvpr/source_fingerprint_cache_scope_v1.json',
    'configs/trackocd_cvpr/M_fingerprint_cache_controller_shadow_v1.json',
    'scripts/trackocd_cvpr/shadow_M_fingerprint_cache.py',
    SHADOW + '/launch.json', SHADOW + '/result.json', SHADOW + '/execution_profile.json',
    SHADOW + '_supervisor_attempt_1.json', FAILED)))


def fingerprint_configuration(root, original, args):
    cfg = json.loads((root / CONFIG).read_text())
    if (cfg['status'] != 'REGISTERED_EQUIVALENT_FINGERPRINT_CACHE_FULL_M_DEV_ACTUAL_ALL11_SHADOW_REQUIRED'
            or cfg['original_config'] != 'configs/trackocd_cvpr/M_GT_causal_development_v1.json'
            or cfg['output_prefix'] != 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_fingerprint_cache_v1'
            or cfg['cache_profile'] != PROFILE
            or cfg['change_data_models_orders_prefixes_thresholds_or_resource_guards']
            or cfg['partial_old_Mean_metric_reuse_model_selection_or_full_Goal_acceptance']
            or not cfg['qualification_is_policy_parity_not_full_development_parity_or_speedup_guarantee']):
        raise ValueError('Exact unchanged full M development, no partial acceptance or protocol change')
    if sha256_file(root / cfg['original_config']) != '2943bdd9c48646bfff277b89b4cc04cbb1f791d9ce5bb057d896a31d69a53f88':
        raise ValueError('Unchanged frozen full M data/method/clock/resource config')
    out = dict(original); out['output_prefix'] = cfg['output_prefix']
    trained = args.model in ('relative_quarantine', 'relative_weighted', 'pooled_weighted')
    out['output_directory'] = out['output_prefix'] + '_' + args.model + ('_s' + str(args.seed) if trained else '')
    return out


def validate_fingerprint_records(result, terminal, execution, reference_cases):
    """Pure public-record checks; file hashes/source/authority gates elsewhere."""
    names = ['Mean_latest', 'StateEMA_latest'] + [v + '_s' + str(s)
        for v in ('relative_quarantine', 'relative_weighted', 'pooled_weighted') for s in (1027, 1028, 1029)]
    counters = (terminal['wall_seconds_including_registration'], terminal['peak_sampled_owned_worker_RSS_bytes'],
        result['resources']['wall_seconds_excluding_registration'], result['resources']['peak_guarded_RSS_bytes'])
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in counters):
        raise ValueError('Actual finite nonnegative execution counters required')
    if (result['status'] != 'ACTUAL_ALL11_FULL_POLICY_RAW_CACHE_CONTROLLER_SHADOW_NOT_FULL_DEV_OR_METHOD_GAIN'
            or set(result['cases']) != set(names) or set(reference_cases) != set(names)
            or (result['physical_tracks_per_case'], result['observations_per_replay']) != (54, 1268)
            or result['optimizer_steps'] != 0 or result['full72development_or_Pred_purity_or_Goal_accepted']
            or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None or not terminal['complete_result_present']
            or not terminal['only_new_owned_Popen_worker_managed']
            or terminal['wall_seconds_including_registration'] > 3000
            or terminal['peak_sampled_owned_worker_RSS_bytes'] > 3221225472
            or result['resources']['CUDA_initialized'] or result['resources']['training_GPU_hours'] != 0
            or result['resources']['peak_guarded_RSS_bytes'] > 3221225472
            or result['resources']['wall_seconds_excluding_registration'] > 3000):
        raise ValueError('Actual whole ALL11 CPU replay/normal completion/bounds required')
    records = execution['ordered_all11_scope_counters']
    if (execution['profile'] != PROFILE or len(records) != 11
            or not execution['global_hashlib_original_sources_data_models_thresholds_orders_and_guards_unchanged']
            or execution['full72_development_speedup_purity_or_Goal_accepted']):
        raise ValueError('Actual explicit complete profile, not source-only or speedup acceptance')
    for name, counters in zip(names, records):
        row = result['cases'][name]
        if any(type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0
               for key in ('old_seconds', 'cache_seconds')):
            raise ValueError('Actual finite paired timing counters required')
        if (not row['actual_full_policy_event_score_routing_chosen_version_numeric_ledger_and_outcome_exact']
                or row['private_trace_sha256'] != reference_cases[name]['private_trace_sha256']
                or not row['timing_is_shared_host_sequential_pair_not_isolated_speedup_claim']
                or any(type(v) is not int or v < 0 for v in counters.values())
                or set(counters) != {'Raw_statistics_bytes', 'fingerprint_bytes', 'fingerprint_hits', 'fingerprint_misses'}
                or counters['Raw_statistics_bytes'] > 33554432 or counters['fingerprint_bytes'] > 33554432
                or row['retained_derived_cache_bytes'] != counters['Raw_statistics_bytes'] + counters['fingerprint_bytes']):
            raise ValueError('Every complete actual scope/trace/exact profile byte bound required')
        if name in ('Mean_latest', 'StateEMA_latest') and not counters['fingerprint_hits']:
            raise ValueError('Actual baseline fingerprint cache exercised, not a no-op')


def verify_fingerprint_shadow(root):
    cfg = json.loads((root / CONFIG).read_text())
    for key in ('shadow_result_sha256', 'shadow_execution_profile_sha256'):
        if not isinstance(cfg.get(key), str) or re.fullmatch('[0-9a-f]{64}', cfg[key]) is None:
            raise ValueError('Actual entire ALL11 shadow/profile fingerprints pending')
    raw = verify_actual_shadow(root)
    if (cfg['shadow_directory'] != SHADOW or cfg['cached_failed_terminal'] != FAILED
            or cfg['cached_failed_terminal_sha256'] != '04d07895d893c67b3fd6ef7a2b50ec0263f72156d3ab2f4cc88a9839819ab5e9'
            or cfg['shadow_preregistration_commit'] != 'e35fca60ef3e8eb94bbf5f994da95ed0d634feac'):
        raise ValueError('Exact new actual shadow/original cached failure binding')
    for path, digest in ((SHADOW + '/result.json', cfg['shadow_result_sha256']),
                         (SHADOW + '/execution_profile.json', cfg['shadow_execution_profile_sha256']),
                         (FAILED, cfg['cached_failed_terminal_sha256'])):
        if (root / path).is_symlink() or sha256_file(root / path) != digest:
            raise ValueError('Unchanged complete public proof fingerprints required')
    result = json.loads((root / (SHADOW + '/result.json')).read_text())
    terminal = json.loads((root / (SHADOW + '_supervisor_attempt_1.json')).read_text())
    execution = json.loads((root / (SHADOW + '/execution_profile.json')).read_text())
    failed = json.loads((root / FAILED).read_text())
    launch = json.loads((root / (SHADOW + '/launch.json')).read_text())
    for record in (result, launch):
        if record['live_remote_registration_proof']['sha'] != cfg['shadow_preregistration_commit']:
            raise ValueError('Actual fixed shadow preregistration, not a replacement selected result')
    for path in ('src/trackocd_cvpr/qualified_fingerprint_cache_scope.py',
                 'configs/trackocd_cvpr/source_fingerprint_cache_scope_v1.json'):
        if launch['source_sha256'][path] != sha256_file(root / path):
            raise ValueError('Exact qualified own-process profile source, not changed kernel')
    if (execution['result_sha256'] != cfg['shadow_result_sha256'] or failed['successful_owned_execution']
            or failed['guard_stop_reason'] != 'owned_job_wall' or failed['complete_result_present']):
        raise ValueError('Full profile-result binding and unchanged cached Mean failure')
    reference = json.loads((root / (OLD_SHADOW + '/result.json')).read_text())['cases']
    validate_fingerprint_records(result, terminal, execution, reference)
    return {**raw, 'actual_entire_ALL11_fingerprint_shadow_profile_checked': True,
        'fingerprint_shadow_result_sha256': cfg['shadow_result_sha256'],
        'fingerprint_shadow_execution_profile_sha256': cfg['shadow_execution_profile_sha256'],
        'cached_failed_terminal_sha256': cfg['cached_failed_terminal_sha256'],
        'full_development_parity_speedup_or_partial_old_metric_reuse_claim': False}
