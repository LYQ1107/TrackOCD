"""Public completed-shadow gate for a namespace-only cached full M replay."""
import json
import re
from src.trackocd_v2.io import sha256_file

CONFIG = 'configs/trackocd_cvpr/M_GT_causal_development_raw_cache_v1.json'
SHADOW = 'outputs/trackocd_cvpr/cvpr4/M_raw_cache_controller_shadow_v1'
NUMERIC = 'outputs/trackocd_cvpr/cvpr4/qualified_frozen_raw_statistics_v1/result.json'
FAILED = 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_v1_Mean_latest_supervisor_attempt_1.json'
SOURCES = (CONFIG, 'src/trackocd_cvpr/M_cached_development_profile.py',
           'src/trackocd_cvpr/qualified_raw_cache_scope.py', 'src/trackocd_cvpr/frozen_raw_statistics_cache.py',
           SHADOW + '/launch.json', SHADOW + '/result.json', SHADOW + '_supervisor_attempt_1.json', NUMERIC, FAILED)


def cached_configuration(root, original, args):
    profile = json.loads((root / CONFIG).read_text())
    if (profile['status'] != 'REGISTERED_EQUIVALENT_RAW_CACHE_FULL_M_DEV_ACTUAL_SHADOW_REQUIRED'
            or profile['original_config'] != 'configs/trackocd_cvpr/M_GT_causal_development_v1.json'
            or profile['output_prefix'] != 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_raw_cache_v1'
            or profile['change_labels_orders_seeds_weights_thresholds_prefixes_guards_or_coverage']
            or profile['reuse_partial_original_Mean_metrics_or_model_selection']
            or profile['full_development_result_or_Goal_acceptance_from_config']):
        raise ValueError('Only explicit sameprotocol cached fullreplay, no selected partial results')
    if sha256_file(root / profile['original_config']) != '2943bdd9c48646bfff277b89b4cc04cbb1f791d9ce5bb057d896a31d69a53f88':
        raise ValueError('Exact original frozen M development config, not changed input/guard protocol')
    out = dict(original); out['output_prefix'] = profile['output_prefix']
    trained = args.model in ('relative_quarantine', 'relative_weighted', 'pooled_weighted')
    out['output_directory'] = out['output_prefix'] + '_' + args.model + ('_s' + str(args.seed) if trained else '')
    return out


def verify_actual_shadow(root):
    p = json.loads((root / CONFIG).read_text())
    expected = p['shadow_result_sha256']
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ValueError('Actual shadow fingerprint pending; no real replay from source tests')
    if p['shadow_directory'] != SHADOW or p['numeric_qualification'] != NUMERIC or p['original_failed_terminal'] != FAILED:
        raise ValueError('Exact qualified scope/old failure paths')
    if p['numeric_qualification_sha256'] != '35743611e5eb395bd4091fac50b9e815515a6404c2f95ad5328c594580144f57' or p['original_failed_terminal_sha256'] != '3fd41182dca48bc873cfcba7ff61e46849402bd4cd27458a2a86a6a8819315d1':
        raise ValueError('Exact actual numeric qualification/old failure binding')
    for path, digest in ((SHADOW + '/result.json', expected), (NUMERIC, p['numeric_qualification_sha256']), (FAILED, p['original_failed_terminal_sha256'])):
        if (root / path).is_symlink() or sha256_file(root / path) != digest:
            raise ValueError('Changed/redirected public qualification')
    r = json.loads((root / (SHADOW + '/result.json')).read_text())
    t = json.loads((root / (SHADOW + '_supervisor_attempt_1.json')).read_text())
    failed = json.loads((root / FAILED).read_text())
    if (r['status'] != 'ACTUAL_ALL11_FULL_POLICY_RAW_CACHE_CONTROLLER_SHADOW_NOT_FULL_DEV_OR_METHOD_GAIN'
            or not t['successful_owned_execution'] or t['owned_worker_exit_code'] != 0 or t['guard_stop_reason'] is not None
            or not t['complete_result_present'] or r['optimizer_steps'] != 0
            or r['full72development_or_Pred_purity_or_Goal_accepted']
            or failed['successful_owned_execution'] or failed['guard_stop_reason'] != 'owned_job_wall'
            or failed['complete_result_present']):
        raise ValueError('Actual successful all11 pairedshadow and unchanged oldfailure required')
    cases = {'Mean_latest', 'StateEMA_latest'} | {v + '_s' + str(s) for v in ('relative_quarantine', 'relative_weighted', 'pooled_weighted') for s in (1027, 1028, 1029)}
    if (set(r['cases']) != cases or r['physical_tracks_per_case'] != 54 or r['observations_per_replay'] != 1268
            or any(not row['actual_full_policy_event_score_routing_chosen_version_numeric_ledger_and_outcome_exact']
                   or row['retained_derived_cache_bytes'] > 67108864 for row in r['cases'].values())
            or r['resources']['CUDA_initialized'] or r['resources']['peak_guarded_RSS_bytes'] > 3221225472
            or r['resources']['wall_seconds_excluding_registration'] > 3000
            or t['peak_sampled_owned_worker_RSS_bytes'] > 3221225472 or t['wall_seconds_including_registration'] > 3000):
        raise ValueError('All full-policy modelcases/actualequality/bounds, no successful subset')
    return {'actual_all11_policy_controller_shadow_checked': True,
            'shadow_result_sha256': expected, 'numeric_qualification_sha256': p['numeric_qualification_sha256'],
            'old_failed_terminal_sha256': p['original_failed_terminal_sha256'],
            'original_full72_Mean_parity_or_partial_metric_reuse_claim': False}
