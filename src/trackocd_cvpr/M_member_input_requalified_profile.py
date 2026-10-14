"""Fail-closed gate for the source-bound-requalified ALL11 input shadow."""
import json
import re

from src.trackocd_v2.io import sha256_file
from scripts.trackocd_cvpr.shadow_M_member_feature_cache_support_requalification import (
    CONFIG as REQUAL_CONFIG, PRIOR, configuration_gate)

SHADOW = 'outputs/trackocd_cvpr/cvpr4/M_member_feature_cache_shadow_support_requalification_v1'
BINDING = SHADOW + '/requalification_binding.json'
PROFILE = 'RAW32MiB_SHA15.5MiB_MEMBER_INPUT16MiB_TEMP_INDEX0.5MiB_TOTAL64MiB'
SOURCES = tuple(dict.fromkeys((
    REQUAL_CONFIG, 'scripts/trackocd_cvpr/shadow_M_member_feature_cache_support_requalification.py',
    'src/trackocd_cvpr/M_member_input_requalified_profile.py', PRIOR,
    SHADOW + '/launch.json', SHADOW + '/result.json', SHADOW + '/execution_profile.json',
    SHADOW + '/requalification_binding.json', SHADOW + '_supervisor_attempt_1.json')))


def verify_requalified_member_input_shadow(root):
    cfg = json.loads((root / REQUAL_CONFIG).read_text())
    configuration_gate(root, cfg)
    if cfg.get('binding_path') != BINDING:
        raise ValueError('Exact requalification binding path required')
    binding = json.loads((root / BINDING).read_text())
    required = {'preregistration_commit', 'result_sha256', 'execution_profile_sha256', 'terminal_sha256'}
    if set(binding) != required or any(not isinstance(binding[key], str) for key in required):
        raise ValueError('Complete requalification binding required')
    if (re.fullmatch('[0-9a-f]{40}', binding['preregistration_commit']) is None
            or any(re.fullmatch('[0-9a-f]{64}', binding[key]) is None
                   for key in ('result_sha256', 'execution_profile_sha256', 'terminal_sha256'))):
        raise ValueError('Requalification binding fingerprints malformed')
    paths = {
        SHADOW + '/result.json': binding['result_sha256'],
        SHADOW + '/execution_profile.json': binding['execution_profile_sha256'],
        SHADOW + '_supervisor_attempt_1.json': binding['terminal_sha256'],
    }
    for path, digest in paths.items():
        p = root / path
        if p.is_symlink() or not p.is_file() or sha256_file(p) != digest:
            raise ValueError('Requalification artifact binding mismatch: ' + path)
    result = json.loads((root / (SHADOW + '/result.json')).read_text())
    profile = json.loads((root / (SHADOW + '/execution_profile.json')).read_text())
    terminal = json.loads((root / (SHADOW + '_supervisor_attempt_1.json')).read_text())
    launch = json.loads((root / (SHADOW + '/launch.json')).read_text())
    previous = json.loads((root / PRIOR).read_text())
    if (launch['live_remote_registration_proof']['sha'] != binding['preregistration_commit']
            or result['live_remote_registration_proof']['sha'] != binding['preregistration_commit']
            or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None or not terminal['complete_result_present']
            or terminal['training_GPU_hours'] != 0
            or terminal['peak_sampled_owned_worker_RSS_bytes'] > 3221225472
            or terminal['wall_seconds_including_registration'] > 3000
            or result['optimizer_steps'] != 0
            or result['full72development_or_Pred_purity_or_Goal_accepted']
            or profile['profile'] != PROFILE
            or profile['result_sha256'] != binding['result_sha256']
            or not profile['actual_all11_trace_digests_match_previous_profile']
            or not profile['source_bound_requalification']
            or profile['source_bound_after_sha256'] != sha256_file(root / 'src/trackocd_cvpr/member_reliability.py')
            or profile['learned_output_memo_or_goal_acceptance']):
        raise ValueError('Complete current-source ALL11 shadow, not partial/accepted science')
    names = ['Mean_latest', 'StateEMA_latest'] + [v + '_s' + str(s)
        for v in ('relative_quarantine', 'relative_weighted', 'pooled_weighted')
        for s in (1027, 1028, 1029)]
    if set(result['cases']) != set(names) or len(profile['ordered_all11_scope_counters']) != 11:
        raise ValueError('All eleven shadow cases and scopes required')
    for name, row in zip(names, profile['ordered_all11_scope_counters']):
        case = result['cases'][name]
        if (case['private_trace_sha256'] != previous['cases'][name]['private_trace_sha256']
                or not case['actual_full_policy_event_score_routing_chosen_version_numeric_ledger_and_outcome_exact']
                or row['fingerprint_hits'] < 1 or row['temporary_index_hits'] < 1
                or (name not in ('Mean_latest', 'StateEMA_latest') and row['member_input_hits'] < 1)
                or row['Raw_statistics_bytes'] > 33554432
                or row['fingerprint_bytes'] > 16252928
                or row['member_input_bytes'] > 16777216
                or row['temporary_index_peak_bookkeeping_bytes'] > 524288):
            raise ValueError('Current-source all11 trace/cache scope parity failed')
    return {
        'actual_ALL11_member_input_and_current_refresh_index_parity_checked': True,
        'member_input_shadow_result_sha256': binding['result_sha256'],
        'member_input_shadow_execution_profile_sha256': binding['execution_profile_sha256'],
        'member_input_shadow_terminal_sha256': binding['terminal_sha256'],
        'source_bound_requalification': True,
        'full_dev_or_speedup_or_Goal_accepted': False,
    }
