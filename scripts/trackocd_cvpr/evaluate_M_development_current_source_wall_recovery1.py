#!/usr/bin/env python3
"""Fresh 6000-second wall recovery for the current-source causal M worker."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import sha256_file
from scripts.trackocd_cvpr import evaluate_M_development_current_source as current

SCRIPT = 'scripts/trackocd_cvpr/evaluate_M_development_current_source_wall_recovery1.py'
CONFIG = 'configs/trackocd_cvpr/M_GT_causal_development_current_source_wall_recovery1.json'
BASE_CONFIG = 'configs/trackocd_cvpr/M_GT_causal_development_current_source_v1.json'
FAILED = 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_current_source_v1_Mean_latest'
RECOVERY_PREFIX = 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_current_source_wall_recovery1'

ORIGINAL_CONFIGURATION = current.configuration
ORIGINAL_SOURCES = current.SOURCES


def configuration(root, args):
    manifest = json.loads((root / CONFIG).read_text())
    expected = {
        'status', 'base_manifest', 'base_manifest_sha256', 'failed_case',
        'failed_namespace', 'failed_launch', 'failed_progress', 'failed_terminal',
        'failed_launch_sha256', 'failed_progress_sha256', 'failed_terminal_sha256',
        'recovery_output_prefix', 'wall_seconds_ceiling',
        'fresh_worker_no_partial_metric_or_checkpoint_reuse',
        'same_current_source_causal_protocol_except_wall_ceiling',
        'no_private_data_weights_or_prediction_history_upload',
        'formal_M_development_or_model_selection',
        'model_selection_or_dev_final_selection',
    }
    if set(manifest) != expected:
        raise ValueError('Exact current-source M wall recovery manifest required')
    if (manifest['status'] != 'REGISTERED_CURRENT_SOURCE_CAUSAL_M_DEV_WALL_RECOVERY1'
            or manifest['base_manifest'] != BASE_CONFIG
            or sha256_file(root / BASE_CONFIG) != manifest['base_manifest_sha256']
            or manifest['failed_namespace'] != FAILED
            or manifest['failed_launch'] != FAILED + '/launch.json'
            or manifest['failed_progress'] != FAILED + '/progress.json'
            or manifest['failed_terminal'] != FAILED + '_supervisor_attempt_1.json'
            or manifest['recovery_output_prefix'] != RECOVERY_PREFIX
            or manifest['wall_seconds_ceiling'] != 6000
            or not manifest['fresh_worker_no_partial_metric_or_checkpoint_reuse']
            or not manifest['same_current_source_causal_protocol_except_wall_ceiling']
            or not manifest['no_private_data_weights_or_prediction_history_upload']
            or not manifest['formal_M_development_or_model_selection']
            or manifest['model_selection_or_dev_final_selection']):
        raise ValueError('Wall recovery widened or changed the frozen causal protocol')
    for path, digest in ((manifest['failed_launch'], manifest['failed_launch_sha256']),
                         (manifest['failed_progress'], manifest['failed_progress_sha256']),
                         (manifest['failed_terminal'], manifest['failed_terminal_sha256'])):
        if sha256_file(root / path) != digest:
            raise ValueError('Preserved wall-failure evidence changed: ' + path)
    progress = json.loads((root / manifest['failed_progress']).read_text())
    terminal = json.loads((root / manifest['failed_terminal']).read_text())
    if (progress.get('completed_order') != 1029 or progress.get('physical_tracks') != 472
            or terminal.get('guard_stop_reason') != 'owned_job_wall'
            or terminal.get('owned_worker_exit_code') != -9
            or terminal.get('complete_result_present')
            or terminal.get('successful_owned_execution')
            or terminal.get('wall_seconds_including_imports_and_network', 0) < 2999):
        raise ValueError('Expected only the preserved 3000-second wall failure')
    cfg = ORIGINAL_CONFIGURATION(root, args)
    cfg['output_prefix'] = RECOVERY_PREFIX
    cfg['output_directory'] = RECOVERY_PREFIX + '_' + args.model + (
        '_s' + str(args.seed) if args.model in current.base.VARIANTS else '')
    cfg['wall_seconds_ceiling'] = manifest['wall_seconds_ceiling']
    return cfg


def _apply_patch():
    current.SCRIPT = SCRIPT
    current.SOURCES = tuple(dict.fromkeys((*ORIGINAL_SOURCES, SCRIPT, CONFIG,
        'src/trackocd_cvpr/current_source_M_development_cohort.py',
        FAILED + '/launch.json', FAILED + '/progress.json', FAILED + '_supervisor_attempt_1.json')))
    current.configuration = configuration


def main():
    _apply_patch()
    return current.main()


if __name__ == '__main__':
    sys.exit(main())
