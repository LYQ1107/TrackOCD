"""Public verifier for the explicit old-endpoint/current-source shadow."""
import json
import re

from src.trackocd_v2.io import sha256_file
from scripts.trackocd_cvpr.shadow_M_member_feature_cache_source_compatibility import (
    CONFIG, PRIOR, configuration_gate)
from .M_endpoint_source_compatibility import CURRENT_SOURCES

SHADOW = 'outputs/trackocd_cvpr/cvpr4/M_member_feature_cache_shadow_source_compatibility_v1'
BINDING = SHADOW + '/source_compatibility_binding.json'
PROFILE = 'RAW32MiB_SHA15.5MiB_MEMBER_INPUT16MiB_TEMP_INDEX0.5MiB_TOTAL64MiB'


def verify_source_compatibility_shadow(root):
    cfg = json.loads((root / CONFIG).read_text())
    configuration_gate(root, cfg)
    binding = json.loads((root / BINDING).read_text())
    required = {'preregistration_commit', 'launch_sha256', 'result_sha256',
                'execution_profile_sha256', 'terminal_sha256', 'source_difference_table_sha256'}
    if set(binding) != required or any(not isinstance(binding[key], str) for key in required):
        raise ValueError('Complete source-compatibility artifact binding required')
    if (re.fullmatch('[0-9a-f]{40}', binding['preregistration_commit']) is None
            or any(re.fullmatch('[0-9a-f]{64}', binding[key]) is None
                   for key in required if key != 'preregistration_commit')):
        raise ValueError('Malformed source-compatibility binding')
    if binding['source_difference_table_sha256'] != '2bbaa92db993effcb813011d7174daf50c9bedeabbba2d063004379ecc19ac10':
        raise ValueError('Unexpected source-difference table')
    paths = {
        SHADOW + '/launch.json': binding['launch_sha256'],
        SHADOW + '/result.json': binding['result_sha256'],
        SHADOW + '/execution_profile.json': binding['execution_profile_sha256'],
        SHADOW + '_supervisor_attempt_1.json': binding['terminal_sha256'],
    }
    for path, digest in paths.items():
        p = root / path
        if p.is_symlink() or not p.is_file() or sha256_file(p) != digest:
            raise ValueError('Source-compatibility artifact mismatch: ' + path)
    launch = json.loads((root / (SHADOW + '/launch.json')).read_text())
    result = json.loads((root / (SHADOW + '/result.json')).read_text())
    profile = json.loads((root / (SHADOW + '/execution_profile.json')).read_text())
    terminal = json.loads((root / (SHADOW + '_supervisor_attempt_1.json')).read_text())
    previous = json.loads((root / PRIOR).read_text())
    if (launch['live_remote_registration_proof']['sha'] != binding['preregistration_commit']
            or result['live_remote_registration_proof']['sha'] != binding['preregistration_commit']
            or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None or not terminal['complete_result_present']
            or terminal['training_GPU_hours'] != 0 or terminal['peak_sampled_owned_worker_RSS_bytes'] > 3221225472
            or terminal['wall_seconds_including_registration'] > 3000
            or result['status'] != 'ACTUAL_ALL11_FULL_POLICY_RAW_CACHE_CONTROLLER_SHADOW_NOT_FULL_DEV_OR_METHOD_GAIN'
            or result['optimizer_steps'] != 0 or result['full72development_or_Pred_purity_or_Goal_accepted']
            or result['resources']['CUDA_initialized'] or result['resources']['training_GPU_hours'] != 0
            or result['resources']['peak_guarded_RSS_bytes'] > 3221225472
            or profile['profile'] != PROFILE or profile['result_sha256'] != binding['result_sha256']
            or not profile['actual_all11_trace_digests_match_previous_profile']
            or not profile['source_compatibility_only']
            or not profile['endpoint_training_source_is_old']
            or profile['current_source_training_provenance_established']
            or not profile['current_source_deployment_trace_compatibility_is_not_M_qualification']):
        raise ValueError('Current-source deployment compatibility shadow is not complete')
    if any(sha256_file(root / path) != digest for path, digest in CURRENT_SOURCES.items()):
        raise ValueError('Current source table changed after compatibility shadow')
    names = ['Mean_latest', 'StateEMA_latest'] + [v + '_s' + str(s)
        for v in ('relative_quarantine', 'relative_weighted', 'pooled_weighted')
        for s in (1027, 1028, 1029)]
    if set(result['cases']) != set(names) or len(profile['ordered_all11_scope_counters']) != 11:
        raise ValueError('All eleven compatibility cases required')
    for name, row in zip(names, profile['ordered_all11_scope_counters']):
        case = result['cases'][name]
        if (case['private_trace_sha256'] != previous['cases'][name]['private_trace_sha256']
                or not case['actual_full_policy_event_score_routing_chosen_version_numeric_ledger_and_outcome_exact']
                or row['fingerprint_hits'] < 1 or row['temporary_index_hits'] < 1
                or (name not in ('Mean_latest', 'StateEMA_latest') and row['member_input_hits'] < 1)):
            raise ValueError('Whole current-source deployment trace/cache parity failed')
    return {
        'actual_all11_current_source_deployment_trace_compatibility_checked': True,
        'result_sha256': binding['result_sha256'],
        'execution_profile_sha256': binding['execution_profile_sha256'],
        'terminal_sha256': binding['terminal_sha256'],
        'current_source_training_provenance_established': False,
        'M_qualification_or_goal_acceptance': False,
    }
