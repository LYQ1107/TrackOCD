"""Configuration gate for one full-M replay under explicit source compatibility."""
import json

from src.trackocd_v2.io import sha256_file
from . import M_member_input_development_profile as member_profile

CONFIG = 'configs/trackocd_cvpr/M_GT_causal_development_source_compatibility_v1.json'
COMPAT = 'outputs/trackocd_cvpr/cvpr4/M_member_feature_cache_shadow_source_compatibility_v1'
FAILED = 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_member_input_memo_v1_relative_quarantine_s1027'
TERMINAL = FAILED + '_supervisor_attempt_1.json'
SOURCES = (
    CONFIG,
    'src/trackocd_cvpr/M_source_compatibility_M_recovery.py',
    'scripts/trackocd_cvpr/evaluate_M_development_source_compatibility.py',
    'src/trackocd_cvpr/M_source_compatibility_shadow_profile.py',
    'src/trackocd_cvpr/M_endpoint_source_compatibility.py',
    COMPAT + '/launch.json', COMPAT + '/result.json', COMPAT + '/execution_profile.json',
    COMPAT + '/source_compatibility_binding.json', COMPAT + '_supervisor_attempt_1.json',
    TERMINAL,
)


def configuration(root, original, args):
    cfg = json.loads((root / CONFIG).read_text())
    expected_keys = {
        'status', 'base_config', 'base_config_sha256', 'original_protocol_config',
        'original_protocol_config_sha256', 'compatibility_shadow_directory',
        'compatibility_shadow_result_sha256', 'compatibility_shadow_profile_sha256',
        'compatibility_shadow_terminal_sha256', 'compatibility_shadow_binding_sha256',
        'failed_original_namespace', 'failed_original_terminal', 'failed_original_terminal_sha256',
        'replay_model', 'replay_seed', 'output_prefix', 'original_wall_seconds_ceiling',
        'replay_wall_seconds_ceiling', 'models_orders_prefixes_thresholds_history_and_metrics_unchanged',
        'only_owned_wall_resource_bound_changed', 'partial_metric_reuse_or_old_namespace_overwrite',
        'optimizer_steps', 'training_GPU_hours', 'current_source_training_provenance_established',
        'M_qualification_or_model_selection_or_goal_acceptance',
        'compatibility_replay_is_not_formal_M_qualification'}
    if set(cfg) != expected_keys:
        raise ValueError('Exact source-compatibility M replay manifest required')
    if (cfg['status'] != 'REGISTERED_SOURCE_COMPATIBILITY_M_FULL_CAUSAL_REPLAY_NOT_QUALIFICATION'
            or cfg['base_config'] != member_profile.CONFIG
            or cfg['base_config_sha256'] != sha256_file(root / member_profile.CONFIG)
            or cfg['original_protocol_config_sha256'] != sha256_file(root / cfg['original_protocol_config'])
            or cfg['compatibility_shadow_directory'] != COMPAT
            or cfg['replay_model'] != 'relative_quarantine' or cfg['replay_seed'] != 1027
            or cfg['failed_original_namespace'] != FAILED or cfg['failed_original_terminal'] != TERMINAL
            or cfg['failed_original_terminal_sha256'] != sha256_file(root / TERMINAL)
            or cfg['original_wall_seconds_ceiling'] != 3000
            or cfg['replay_wall_seconds_ceiling'] != 6000
            or not cfg['models_orders_prefixes_thresholds_history_and_metrics_unchanged']
            or not cfg['only_owned_wall_resource_bound_changed']
            or cfg['partial_metric_reuse_or_old_namespace_overwrite']
            or cfg['optimizer_steps'] != 0 or cfg['training_GPU_hours'] != 0
            or cfg['current_source_training_provenance_established']
            or cfg['M_qualification_or_model_selection_or_goal_acceptance']
            or not cfg['compatibility_replay_is_not_formal_M_qualification']):
        raise ValueError('Only fixed compatibility replay with explicit non-qualification is allowed')
    for path, digest in (
            (COMPAT + '/result.json', cfg['compatibility_shadow_result_sha256']),
            (COMPAT + '/execution_profile.json', cfg['compatibility_shadow_profile_sha256']),
            (COMPAT + '_supervisor_attempt_1.json', cfg['compatibility_shadow_terminal_sha256']),
            (COMPAT + '/source_compatibility_binding.json', cfg['compatibility_shadow_binding_sha256'])):
        if (root / path).is_symlink() or sha256_file(root / path) != digest:
            raise ValueError('Bound compatibility shadow changed: ' + path)
    old_terminal = json.loads((root / TERMINAL).read_text())
    if (old_terminal['guard_stop_reason'] != 'owned_job_wall'
            or old_terminal['owned_worker_exit_code'] != -9
            or old_terminal['complete_result_present']
            or old_terminal['successful_owned_execution']):
        raise ValueError('Original incomplete wall failure must remain preserved')
    base = member_profile.configuration(root, original, args)
    out = dict(base)
    out['output_prefix'] = cfg['output_prefix']
    out['output_directory'] = cfg['output_prefix'] + '_' + args.model + '_s' + str(args.seed)
    out['wall_seconds_ceiling'] = cfg['replay_wall_seconds_ceiling']
    return out
