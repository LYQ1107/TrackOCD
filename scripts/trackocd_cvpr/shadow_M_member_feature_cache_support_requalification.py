#!/usr/bin/env python3
"""Requalify the ALL11 input-only shadow after a source-bound-only change."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import shadow_M_member_feature_cache as base
from scripts.trackocd_cvpr import shadow_M_raw_cache as runtime
from src.trackocd_v2.io import atomic_json, sha256_file

SCRIPT = 'scripts/trackocd_cvpr/shadow_M_member_feature_cache_support_requalification.py'
CONFIG = 'configs/trackocd_cvpr/M_member_feature_cache_shadow_support_requalification_v1.json'
PRIOR = 'outputs/trackocd_cvpr/cvpr4/M_member_feature_cache_shadow_v1/result.json'
OLD_CONFIG = 'configs/trackocd_cvpr/M_member_feature_cache_shadow_v1.json'
FAILED = 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_all_model_fingerprints_v1_relative_quarantine_s1027_RAM_retry1_supervisor_attempt_1.json'
SOURCES = tuple(dict.fromkeys((*base.SOURCES, SCRIPT, CONFIG, OLD_CONFIG, PRIOR, FAILED,
    'outputs/trackocd_cvpr/cvpr4/M_member_feature_cache_shadow_v1/execution_profile.json',
    'outputs/trackocd_cvpr/cvpr4/M_member_feature_cache_shadow_v1_supervisor_attempt_1.json')))


def configuration_gate(root, cfg):
    manifest = json.loads((root / CONFIG).read_text())
    old = json.loads((root / OLD_CONFIG).read_text())
    expected = dict(old)
    expected['output_directory'] = manifest['output_directory']
    for key, value in manifest.items():
        if key not in expected and key not in {
                'status', 'requalification_status', 'base_config', 'base_config_sha256', 'prior_result',
                'prior_result_sha256', 'prior_execution_profile_sha256',
                'prior_terminal_sha256', 'preserved_full_M_failure_terminal_sha256',
                'changed_source', 'changed_source_before_sha256',
                'changed_source_after_sha256', 'changed_source_commit',
                'binding_path',
                'protocol_models_orders_inputs_metrics_and_resource_guards_unchanged',
                'old_shadow_or_failure_overwritten', 'learned_output_memo_or_goal_acceptance'}:
            raise ValueError('Unexpected requalification field')
    if (manifest['status'] != 'REGISTERED_FULL_POLICY_CONTROLLER_RAW_CACHE_SHADOW_ALL9_ENDPOINTS_REQUIRED'
            or manifest['requalification_status'] != 'REGISTERED_FULL_POLICY_MEMBER_INPUT_SHADOW_SUPPORT_BOUND_REQUALIFICATION'
            or manifest['base_config'] != OLD_CONFIG
            or manifest['base_config_sha256'] != sha256_file(root / OLD_CONFIG)
            or manifest['prior_result'] != PRIOR
            or manifest['prior_result_sha256'] != sha256_file(root / PRIOR)
            or manifest['changed_source'] != 'src/trackocd_cvpr/member_reliability.py'
            or manifest['changed_source_after_sha256'] != sha256_file(root / manifest['changed_source'])
            or manifest['changed_source_before_sha256'] != '1ec246809fe85c25209dd175953889339e55ca412a3b767347f5d89d0ac652b8'
            or manifest['changed_source_commit'] != '3583d6a'
            or not manifest['protocol_models_orders_inputs_metrics_and_resource_guards_unchanged']
            or manifest['old_shadow_or_failure_overwritten']
            or manifest['learned_output_memo_or_goal_acceptance']):
        raise ValueError('Only the recorded member-reliability source-bound change may be requalified')
    for key, value in old.items():
        if key == 'output_directory':
            if manifest[key] != expected[key]:
                raise ValueError('New output namespace required')
        elif manifest.get(key) != value:
            raise ValueError('All original shadow protocol fields and guards must remain unchanged: ' + key)
    if sha256_file(root / 'outputs/trackocd_cvpr/cvpr4/M_member_feature_cache_shadow_v1/execution_profile.json') != manifest['prior_execution_profile_sha256']:
        raise ValueError('Prior shadow profile changed')
    if sha256_file(root / 'outputs/trackocd_cvpr/cvpr4/M_member_feature_cache_shadow_v1_supervisor_attempt_1.json') != manifest['prior_terminal_sha256']:
        raise ValueError('Prior shadow terminal changed')
    if sha256_file(root / FAILED) != manifest['preserved_full_M_failure_terminal_sha256']:
        raise ValueError('Preserved full-M failure changed')
    prior = json.loads((root / PRIOR).read_text())
    if (prior['status'] != 'ACTUAL_ALL11_FULL_POLICY_RAW_CACHE_CONTROLLER_SHADOW_NOT_FULL_DEV_OR_METHOD_GAIN'
            or len(prior['cases']) != 11 or prior['optimizer_steps'] != 0
            or prior['full72development_or_Pred_purity_or_Goal_accepted']):
        raise ValueError('Prior shadow is not the complete non-acceptance reference')
    return prior


def main():
    base.SCRIPT = SCRIPT
    base.CONFIG = CONFIG
    base.PRIOR = PRIOR
    base.SOURCES = SOURCES
    original_gate = runtime.source_gate

    def gate(args, cfg, safety):
        configuration_gate(ROOT, cfg)
        return original_gate(args, cfg, safety)

    base.configuration_gate = configuration_gate
    base.configuration_gate = configuration_gate
    runtime.source_gate = gate
    try:
        result = base.main()
        if '--worker' in sys.argv:
            out = ROOT / json.loads((ROOT / CONFIG).read_text())['output_directory']
            profile_path = out / 'execution_profile.json'
            profile = json.loads(profile_path.read_text())
            profile.update({
                'source_bound_requalification': True,
                'source_bound_change_path': 'src/trackocd_cvpr/member_reliability.py',
                'source_bound_before_sha256': '1ec246809fe85c25209dd175953889339e55ca412a3b767347f5d89d0ac652b8',
                'source_bound_after_sha256': sha256_file(ROOT / 'src/trackocd_cvpr/member_reliability.py'),
                'old_shadow_or_failure_overwritten': False,
                'learned_output_memo_or_goal_acceptance': False,
                'current_source_bound_requalification_is_not_full_development_or_method_gain': True,
            })
            atomic_json(profile_path, profile)
        return result
    finally:
        runtime.source_gate = original_gate


if __name__ == '__main__':
    sys.exit(main())
