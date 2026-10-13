"""Fail-closed prospective full-M gate; pending ALL11 fingerprints refuse."""
import json
import re
from types import SimpleNamespace
from src.trackocd_v2.io import sha256_file
from .M_all_model_fingerprint_development_profile import verify_all_model_shadow, SOURCES as OLD_SOURCES
from scripts.trackocd_cvpr.shadow_M_member_feature_cache import (
    SOURCES as SHADOW_SOURCES, CONFIG as SHADOW_CONFIG, configuration_gate, PROFILE, FAILED, PRIOR)

CONFIG = 'configs/trackocd_cvpr/M_GT_causal_development_member_input_memo_v1.json'
SHADOW = 'outputs/trackocd_cvpr/cvpr4/M_member_feature_cache_shadow_v1'
PREREG = '7a3d9ec7a29840ff5bf425413a004d54f1c35e38'
SOURCES = tuple(dict.fromkeys((*OLD_SOURCES, *SHADOW_SOURCES, CONFIG,
    'src/trackocd_cvpr/M_member_input_development_profile.py',
    'scripts/trackocd_cvpr/evaluate_M_development_member_input_memo.py',
    SHADOW + '/launch.json', SHADOW + '/result.json', SHADOW + '/execution_profile.json',
    SHADOW + '_supervisor_attempt_1.json')))


def configuration(root, original, args):
    cfg = json.loads((root / CONFIG).read_text())
    keys = {'status', 'original_config', 'output_prefix', 'shadow_directory',
        'shadow_preregistration_commit', 'shadow_result_sha256', 'shadow_execution_profile_sha256',
        'shadow_terminal_sha256', 'cache_profile', 'preserved_full_M_RAM_retry_terminal_sha256',
        'models_orders_prefixes_thresholds_history_or_guards_changed',
        'partial_metric_reuse_model_selection_or_full_Goal_acceptance',
        'policy_parity_is_not_full_development_parity_or_speedup_guarantee'}
    if (set(cfg) != keys or cfg['status'] != 'REGISTERED_FROZEN_M_FULL_DEV_ACTUAL_MEMBER_INPUT_ALL11_REQUIRED'
            or cfg['original_config'] != 'configs/trackocd_cvpr/M_GT_causal_development_v1.json'
            or cfg['output_prefix'] != 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_member_input_memo_v1'
            or cfg['shadow_directory'] != SHADOW or cfg['shadow_preregistration_commit'] != PREREG
            or cfg['cache_profile'] != PROFILE
            or cfg['preserved_full_M_RAM_retry_terminal_sha256'] != '97b6ee70217340a4a4dd2a094469343ef6a74f03d25995f81bf09c9abf3193da'
            or cfg['models_orders_prefixes_thresholds_history_or_guards_changed']
            or cfg['partial_metric_reuse_model_selection_or_full_Goal_acceptance']
            or not cfg['policy_parity_is_not_full_development_parity_or_speedup_guarantee']
            or args.model not in ('relative_quarantine', 'relative_weighted', 'pooled_weighted')
            or args.seed not in (1027, 1028, 1029)):
        raise ValueError('Same complete frozen M protocol, qualified input memo/new namespace only')
    if sha256_file(root / cfg['original_config']) != '2943bdd9c48646bfff277b89b4cc04cbb1f791d9ce5bb057d896a31d69a53f88':
        raise ValueError('Unchanged original full72/472/8753/clock/probes/guards')
    out = dict(original)
    out['output_prefix'] = cfg['output_prefix']
    out['output_directory'] = cfg['output_prefix'] + '_' + args.model + '_s' + str(args.seed)
    return out


def verify_member_input_shadow(root):
    configuration(root, {}, SimpleNamespace(model='relative_quarantine', seed=1027))
    cfg = json.loads((root / CONFIG).read_text())
    for key in ('shadow_result_sha256', 'shadow_execution_profile_sha256', 'shadow_terminal_sha256'):
        if not isinstance(cfg.get(key), str) or re.fullmatch('[0-9a-f]{64}', cfg[key]) is None:
            raise ValueError('Actual normally completed ALL11 member-input qualification fingerprints pending')
    configuration_gate(root, json.loads((root / SHADOW_CONFIG).read_text()))
    previous = verify_all_model_shadow(root)
    paths = (SHADOW + '/result.json', SHADOW + '/execution_profile.json', SHADOW + '_supervisor_attempt_1.json')
    for path, key in zip(paths, ('shadow_result_sha256', 'shadow_execution_profile_sha256', 'shadow_terminal_sha256')):
        if (root / path).is_symlink() or sha256_file(root / path) != cfg[key]:
            raise ValueError('Whole actual result/profile/normal terminal fingerprints')
    result, profile, terminal = (json.loads((root / path).read_text()) for path in paths)
    launch = json.loads((root / (SHADOW + '/launch.json')).read_text())
    prior = json.loads((root / PRIOR).read_text())
    names = ['Mean_latest', 'StateEMA_latest'] + [v + '_s' + str(s)
        for v in ('relative_quarantine', 'relative_weighted', 'pooled_weighted') for s in (1027, 1028, 1029)]
    if (result['status'] != 'ACTUAL_ALL11_FULL_POLICY_RAW_CACHE_CONTROLLER_SHADOW_NOT_FULL_DEV_OR_METHOD_GAIN'
            or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None or not terminal['complete_result_present']
            or terminal['training_GPU_hours'] != 0 or terminal['peak_sampled_owned_worker_RSS_bytes'] > 3221225472
            or terminal['wall_seconds_including_registration'] > 3000
            or result['optimizer_steps'] != 0 or result['resources']['CUDA_initialized']
            or result['resources']['training_GPU_hours'] != 0 or result['physical_tracks_per_case'] != 54
            or result['observations_per_replay'] != 1268 or set(result['cases']) != set(names)
            or result['full72development_or_Pred_purity_or_Goal_accepted']
            or profile['profile'] != PROFILE or profile['result_sha256'] != cfg['shadow_result_sha256']
            or not profile['actual_all11_trace_digests_match_previous_profile']
            or profile['learned_logits_prototypes_predictions_or_model_outputs_cached']
            or profile['controller_refreshes_history_members_probes_or_numeric_kernels_changed']
            or profile['full_development_parity_or_speedup_guarantee_or_Goal_accepted']
            or len(profile['ordered_all11_scope_counters']) != 11):
        raise ValueError('Complete normal ALL11 qualification, not partial parity/source/speedup claim')
    for record in (launch, result):
        if record['live_remote_registration_proof']['sha'] != PREREG:
            raise ValueError('Exact original actual shadow preregistration')
    for path in SHADOW_SOURCES:
        if sha256_file(root / path) != launch['source_sha256'].get(path):
            raise ValueError('Actually qualified original source/cache/kernel changed: ' + path)
    for name, row in zip(names, profile['ordered_all11_scope_counters']):
        case = result['cases'][name]
        if (row['case'] != name or not case['actual_full_policy_event_score_routing_chosen_version_numeric_ledger_and_outcome_exact']
                or case['private_trace_sha256'] != prior['cases'][name]['private_trace_sha256']
                or case['retained_derived_cache_bytes'] > 67108864
                or row['Raw_statistics_bytes'] > 33554432 or row['fingerprint_bytes'] > 16252928
                or row['member_input_bytes'] > 16777216 or row['temporary_index_peak_bookkeeping_bytes'] > 524288
                or row['fingerprint_hits'] < 1 or row['temporary_index_hits'] < 1
                or (name not in ('Mean_latest', 'StateEMA_latest') and row['member_input_hits'] < 1)):
            raise ValueError('ALL11 actual whole trace parity and relevant caches exercised within bounds')
    return {**previous, 'actual_ALL11_member_input_and_current_refresh_index_parity_checked': True,
        'member_input_shadow_result_sha256': cfg['shadow_result_sha256'],
        'member_input_shadow_execution_profile_sha256': cfg['shadow_execution_profile_sha256'],
        'full_dev_or_speedup_or_Goal_accepted': False}
