"""Fail-closed actual ALL11 support-SHA profile gate for full frozen-M dev.

Pending result fingerprints refuse before other inputs. A source or partial
shadow is never accepted; requires complete actual paired trace proof, actual
normal owned wait, profile-result binding, unchanged original failure and
kernel/profile fingerprints. Policy parity is not full-dev speedup proof.
"""
import json
import re
from .M_fingerprint_development_profile import (
    verify_fingerprint_shadow, validate_fingerprint_records, PROFILE as OLD_PROFILE,
    SOURCES as OLD_SOURCES, SHADOW as OLD_SHADOW)
from src.trackocd_v2.io import sha256_file

CONFIG = 'configs/trackocd_cvpr/M_GT_causal_development_all_model_fingerprints_v1.json'
SHADOW = 'outputs/trackocd_cvpr/cvpr4/M_all_model_fingerprint_shadow_v1'
PROFILE = 'EXACT_RAW_SUPPORT_SHA_ALL11_32MiB_AND_FROZEN_RAW_32MiB_TOTAL_64MiB'
FAILED = 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_fingerprint_cache_v1_relative_quarantine_s1027_supervisor_attempt_1.json'
FAILED_SHA = '3c4592996c88992fac9cd02c28fe2df38d962b667b735492901d5a45d9ad643e'
PREREG = 'e3671f1e10112d77c10922fa9b69e4a8904f243b'
SOURCES = tuple(dict.fromkeys((*OLD_SOURCES, CONFIG, FAILED,
    'src/trackocd_cvpr/M_all_model_fingerprint_development_profile.py',
    'src/trackocd_cvpr/qualified_M_fingerprint_cache_scope.py',
    'configs/trackocd_cvpr/M_all_model_fingerprint_shadow_v1.json',
    'scripts/trackocd_cvpr/shadow_M_all_model_fingerprints.py',
    SHADOW + '/launch.json', SHADOW + '/result.json', SHADOW + '/execution_profile.json',
    SHADOW + '_supervisor_attempt_1.json')))


def all_model_configuration(root, original, args):
    cfg = json.loads((root / CONFIG).read_text())
    if (set(cfg) != {'status', 'original_config', 'output_prefix', 'shadow_directory',
            'shadow_result_sha256', 'shadow_execution_profile_sha256', 'shadow_preregistration_commit',
            'failed_trained_M_terminal_sha256', 'cache_profile',
            'change_data_models_orders_prefixes_thresholds_or_resource_guards',
            'partial_metric_reuse_model_selection_or_full_Goal_acceptance',
            'qualification_is_policy_parity_not_full_development_parity_or_speedup_guarantee'}
            or cfg['status'] != 'REGISTERED_FULL_TRAINED_M_DEV_ACTUAL_ALL11_SUPPORT_SHA_PARITY_REQUIRED'
            or cfg['original_config'] != 'configs/trackocd_cvpr/M_GT_causal_development_v1.json'
            or cfg['output_prefix'] != 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_all_model_fingerprints_v1'
            or cfg['shadow_directory'] != SHADOW or cfg['shadow_preregistration_commit'] != PREREG
            or cfg['failed_trained_M_terminal_sha256'] != FAILED_SHA or cfg['cache_profile'] != PROFILE
            or cfg['change_data_models_orders_prefixes_thresholds_or_resource_guards']
            or cfg['partial_metric_reuse_model_selection_or_full_Goal_acceptance']
            or not cfg['qualification_is_policy_parity_not_full_development_parity_or_speedup_guarantee']
            or args.model not in ('relative_quarantine', 'relative_weighted', 'pooled_weighted')
            or args.seed not in (1027, 1028, 1029)):
        raise ValueError('Same frozen full M development; only qualified profile/new namespace differs')
    if sha256_file(root / cfg['original_config']) != '2943bdd9c48646bfff277b89b4cc04cbb1f791d9ce5bb057d896a31d69a53f88':
        raise ValueError('Unchanged original full72 development/data/clock/model/guard protocol')
    out = dict(original); out['output_prefix'] = cfg['output_prefix']
    out['output_directory'] = cfg['output_prefix'] + '_' + args.model + '_s' + str(args.seed)
    return out


def validate_all_model_records(result, terminal, execution, reference):
    if (execution['profile'] != PROFILE
            or not execution['actual_all11_trace_digests_match_previous_profile']
            or not execution['global_hashlib_numeric_hashes_models_data_thresholds_orders_history_and_guards_unchanged']
            or execution['full72_development_speedup_purity_or_Goal_accepted']
            or any(type(row.get('fingerprint_hits')) is not int or row['fingerprint_hits'] < 1
                   for row in execution['ordered_all11_scope_counters'])):
        raise ValueError('Actual ALL11 including every weighted-M cache exercised, no changed kernels')
    # Reuse all original whole-scope/finite counters/byte ceilings/normal wait/
    # trace parity checks, with explicit translation of NEW public framing only.
    old_framing = dict(execution)
    old_framing['profile'] = OLD_PROFILE
    old_framing['global_hashlib_original_sources_data_models_thresholds_orders_and_guards_unchanged'] = True
    validate_fingerprint_records(result, terminal, old_framing, reference)


def verify_all_model_shadow(root):
    cfg = json.loads((root / CONFIG).read_text())
    for key in ('shadow_result_sha256', 'shadow_execution_profile_sha256'):
        if not isinstance(cfg.get(key), str) or re.fullmatch('[0-9a-f]{64}', cfg[key]) is None:
            raise ValueError('Actual complete ALL11 support-SHA fingerprints pending')
    if (cfg['shadow_directory'] != SHADOW or cfg['shadow_preregistration_commit'] != PREREG
            or cfg['failed_trained_M_terminal_sha256'] != FAILED_SHA or cfg['cache_profile'] != PROFILE):
        raise ValueError('Exact unchanged ALL11 qualification/failure profile binding')
    old = verify_fingerprint_shadow(root)
    for path, digest in ((SHADOW + '/result.json', cfg['shadow_result_sha256']),
                         (SHADOW + '/execution_profile.json', cfg['shadow_execution_profile_sha256']),
                         (FAILED, FAILED_SHA)):
        if (root / path).is_symlink() or sha256_file(root / path) != digest:
            raise ValueError('Exact whole result/profile and preserved trained-M wall failure')
    result = json.loads((root / (SHADOW + '/result.json')).read_text())
    terminal = json.loads((root / (SHADOW + '_supervisor_attempt_1.json')).read_text())
    execution = json.loads((root / (SHADOW + '/execution_profile.json')).read_text())
    launch = json.loads((root / (SHADOW + '/launch.json')).read_text())
    failed = json.loads((root / FAILED).read_text())
    if (execution['result_sha256'] != cfg['shadow_result_sha256']
            or failed['guard_stop_reason'] != 'owned_job_wall' or failed['owned_worker_exit_code'] != -9
            or failed['successful_owned_execution'] or failed['complete_result_present']):
        raise ValueError('Whole profile-result binding; no partial trained-M metrics or rescued failure')
    for record in (launch, result):
        if record['live_remote_registration_proof']['sha'] != PREREG:
            raise ValueError('Exact actual preregistered ALL11 trial, not selected replacement')
    for path in ('src/trackocd_cvpr/qualified_M_fingerprint_cache_scope.py',
                 'src/trackocd_cvpr/qualified_fingerprint_cache_scope.py',
                 'scripts/trackocd_cvpr/shadow_M_all_model_fingerprints.py',
                 'configs/trackocd_cvpr/M_all_model_fingerprint_shadow_v1.json',
                 'src/trackocd_cvpr/member_reliability.py',
                 'src/trackocd_cvpr/versioned_reliability_memory.py'):
        if launch['source_sha256'][path] != sha256_file(root / path):
            raise ValueError('Exact actually qualified memo and unchanged numerical memory kernels')
    reference = json.loads((root / (OLD_SHADOW + '/result.json')).read_text())['cases']
    validate_all_model_records(result, terminal, execution, reference)
    return {**old, 'actual_entire_ALL11_weighted_M_support_fingerprint_shadow_checked': True,
        'all_model_shadow_result_sha256': cfg['shadow_result_sha256'],
        'all_model_shadow_execution_profile_sha256': cfg['shadow_execution_profile_sha256'],
        'preserved_failed_trained_M_terminal_sha256': FAILED_SHA,
        'full_development_parity_speedup_or_partial_metric_reuse_claim': False}
