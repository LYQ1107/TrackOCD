#!/usr/bin/env python3
"""ALL11 current-source deployment trace audit with explicit old endpoint receipts."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import shadow_M_member_feature_cache as base
from scripts.trackocd_cvpr import shadow_M_raw_cache as runtime
from src.trackocd_cvpr.M_endpoint_source_compatibility import verified_cohort
from src.trackocd_v2.io import atomic_json, sha256_file

SCRIPT = 'scripts/trackocd_cvpr/shadow_M_member_feature_cache_source_compatibility.py'
CONFIG = 'configs/trackocd_cvpr/M_member_feature_cache_shadow_source_compatibility_v1.json'
PRIOR = 'outputs/trackocd_cvpr/cvpr4/M_member_feature_cache_shadow_v1/result.json'
OLD_CONFIG = 'configs/trackocd_cvpr/M_member_feature_cache_shadow_v1.json'
FAILED_REQUAL = 'outputs/trackocd_cvpr/cvpr4/M_member_feature_cache_shadow_support_requalification_v1_supervisor_attempt_1.json'
SOURCES = tuple(dict.fromkeys((*base.SOURCES, SCRIPT, CONFIG,
    'src/trackocd_cvpr/M_endpoint_source_compatibility.py', OLD_CONFIG, PRIOR, FAILED_REQUAL)))


def configuration_gate(root, cfg):
    manifest = json.loads((root / CONFIG).read_text())
    old = json.loads((root / OLD_CONFIG).read_text())
    for key, value in old.items():
        expected = manifest['output_directory'] if key == 'output_directory' else value
        if manifest.get(key) != expected:
            raise ValueError('Original shadow protocol/guard changed: ' + key)
    if (manifest['compatibility_status'] != 'REGISTERED_SOURCE_BOUND_ENDPOINT_DEPLOYMENT_TRACE_AUDIT_NOT_CURRENT_SOURCE_TRAINING_PROVENANCE'
            or manifest['prior_shadow_result'] != PRIOR
            or manifest['prior_shadow_result_sha256'] != sha256_file(root / PRIOR)
            or manifest['preserved_requalification_terminal'] != FAILED_REQUAL
            or sha256_file(root / FAILED_REQUAL) != manifest['preserved_requalification_terminal_sha256']
            or manifest['changed_source'] != 'src/trackocd_cvpr/member_reliability.py'
            or manifest['endpoint_receipt_source_sha256'] != '1ec246809fe85c25209dd175953889339e55ca412a3b767347f5d89d0ac652b8'
            or manifest['current_source_sha256'] != sha256_file(root / manifest['changed_source'])
            or manifest['declared_endpoint_source_paths'] != [
                'scripts/trackocd_cvpr/pilot_memory_optimizer.py',
                'scripts/trackocd_cvpr/qualify_policy_training_input.py',
                'src/trackocd_cvpr/member_reliability.py',
                'src/trackocd_cvpr/prefix_member_bank.py',
                'src/trackocd_cvpr/versioned_reliability_memory.py',
                'src/trackocd_cvpr/memory_transport_recovery.py']
            or manifest['declared_endpoint_source_difference_table_sha256'] != '2bbaa92db993effcb813011d7174daf50c9bedeabbba2d063004379ecc19ac10'
            or not manifest['only_declared_endpoint_receipt_source_differences_allowed']
            or manifest['current_source_training_provenance_established']
            or not manifest['deployment_trace_compatibility_is_not_M_qualification']
            or manifest['old_shadow_or_failure_overwritten']
            or manifest['model_selection_or_goal_acceptance']):
        raise ValueError('Exact source-compatibility-only audit manifest required')
    prior = json.loads((root / PRIOR).read_text())
    if (prior['status'] != 'ACTUAL_ALL11_FULL_POLICY_RAW_CACHE_CONTROLLER_SHADOW_NOT_FULL_DEV_OR_METHOD_GAIN'
            or len(prior['cases']) != 11 or prior['optimizer_steps'] != 0
            or prior['full72development_or_Pred_purity_or_Goal_accepted']):
        raise ValueError('Complete old non-acceptance shadow reference required')
    return prior


def main():
    base.SCRIPT = SCRIPT
    base.CONFIG = CONFIG
    base.PRIOR = PRIOR
    base.SOURCES = SOURCES
    base.configuration_gate = configuration_gate
    original_cohort = runtime.verified_cohort
    runtime.verified_cohort = verified_cohort
    try:
        result = base.main()
        if '--worker' in sys.argv:
            out = ROOT / json.loads((ROOT / CONFIG).read_text())['output_directory']
            profile_path = out / 'execution_profile.json'
            profile = json.loads(profile_path.read_text())
            profile.update({
                'source_compatibility_only': True,
                'endpoint_training_source_is_old': True,
                'current_source_training_provenance_established': False,
                'current_source_deployment_trace_compatibility_is_not_M_qualification': True,
                'current_member_source_sha256': sha256_file(ROOT / 'src/trackocd_cvpr/member_reliability.py'),
            })
            atomic_json(profile_path, profile)
        return result
    finally:
        runtime.verified_cohort = original_cohort


if __name__ == '__main__':
    sys.exit(main())
