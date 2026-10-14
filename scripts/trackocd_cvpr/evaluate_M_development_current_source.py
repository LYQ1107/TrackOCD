#!/usr/bin/env python3
"""Current-source wrapper for the registered read-only causal M development."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import sha256_file
from src.trackocd_cvpr.current_source_endpoint_cohort import audit_cohort
from src.trackocd_cvpr import current_source_M_development_cohort as current_cohort
from scripts.trackocd_cvpr import evaluate_M_development as base

SCRIPT = 'scripts/trackocd_cvpr/evaluate_M_development_current_source.py'
CONFIG = 'configs/trackocd_cvpr/M_GT_causal_development_current_source_v1.json'
BASE_CONFIG = 'configs/trackocd_cvpr/M_GT_causal_development_v1.json'
AUDIT_CONFIG = 'configs/trackocd_cvpr/M_current_source_endpoint_cohort_audit_v1.json'
AUDIT_RESULT = 'outputs/trackocd_cvpr/cvpr4/M_current_source_endpoint_cohort_audit_v1/result.json'

ORIGINAL_CONFIGURATION = base.configuration
ORIGINAL_SOURCES = base.SOURCES
SOURCES = tuple(dict.fromkeys((*ORIGINAL_SOURCES, SCRIPT, CONFIG, BASE_CONFIG,
    'src/trackocd_cvpr/current_source_M_development_cohort.py',
    'src/trackocd_cvpr/current_source_endpoint_cohort.py',
    'scripts/trackocd_cvpr/verify_current_source_endpoint_cohort.py',
    AUDIT_CONFIG, AUDIT_RESULT, *current_cohort.cohort_sources(ROOT))))


def configuration(root, args):
    manifest = json.loads((root / CONFIG).read_text())
    if set(manifest) != {
            'status', 'base_protocol', 'base_protocol_sha256', 'endpoint_cohort_manifest',
            'endpoint_cohort_manifest_sha256', 'endpoint_cohort_audit_result',
            'endpoint_cohort_audit_result_sha256', 'output_prefix', 'execution_device',
            'CPU_compute_workers', 'optimizer_steps', 'select_model_or_checkpoint',
            'model_selection_or_dev_final_selection',
            'final_Val_Test_encoder_externalweights_newNonKnown_truth_or_vocabulary',
            'no_private_data_weights_or_prediction_history_upload',
            'formal_M_development_is_registered_but_not_Goal_acceptance',
            'all_nine_current_source_endpoints_required_before_private_GT'}:
        raise ValueError('Exact current-source M development manifest required')
    if (manifest['status'] != 'REGISTERED_CURRENT_SOURCE_READONLY_CAUSAL_M_DEV_ALL9_AUDITED_ENDPOINTS_REQUIRED'
            or manifest['base_protocol'] != BASE_CONFIG
            or sha256_file(root / BASE_CONFIG) != manifest['base_protocol_sha256']
            or sha256_file(root / manifest['endpoint_cohort_manifest']) != manifest['endpoint_cohort_manifest_sha256']
            or sha256_file(root / manifest['endpoint_cohort_audit_result']) != manifest['endpoint_cohort_audit_result_sha256']
            or manifest['execution_device'] != 'CPU' or manifest['CPU_compute_workers'] != 1
            or manifest['optimizer_steps'] != 0 or manifest['select_model_or_checkpoint']
            or manifest['model_selection_or_dev_final_selection']
            or manifest['final_Val_Test_encoder_externalweights_newNonKnown_truth_or_vocabulary']
            or not manifest['no_private_data_weights_or_prediction_history_upload']
            or not manifest['formal_M_development_is_registered_but_not_Goal_acceptance']
            or not manifest['all_nine_current_source_endpoints_required_before_private_GT']):
        raise ValueError('Current-source M development manifest changed or widened')
    audit = json.loads((root / manifest['endpoint_cohort_audit_result']).read_text())
    if audit['status'] != 'COHORT_AUDIT_PASS_CURRENT_SOURCE_ENDPOINT_PROVENANCE_ONLY':
        raise ValueError('Current-source endpoint cohort audit is not a pass')
    state = base.CONFIG
    base.CONFIG = BASE_CONFIG
    try:
        cfg = ORIGINAL_CONFIGURATION(root, args)
    finally:
        base.CONFIG = state
    cfg['output_prefix'] = manifest['output_prefix']
    cfg['output_directory'] = manifest['output_prefix'] + '_' + args.model + (
        '_s' + str(args.seed) if args.model in base.VARIANTS else '')
    cfg['current_source_endpoint_cohort_manifest'] = manifest['endpoint_cohort_manifest']
    cfg['current_source_endpoint_cohort_audit_result'] = manifest['endpoint_cohort_audit_result']
    return cfg


def _apply_patch():
    base.CONFIG = BASE_CONFIG
    base.SCRIPT = SCRIPT
    base.SOURCES = SOURCES
    base.cohort_sources = lambda: current_cohort.cohort_sources(ROOT)
    base.verified_cohort = lambda root: current_cohort.verified_cohort(root)
    base.configuration = configuration


def main():
    _apply_patch()
    return base.main()


if __name__ == '__main__':
    sys.exit(main())
