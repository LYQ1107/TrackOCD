"""Fail-closed registration for the evidence-driven full-M wall recovery.

The prior member-input attempt completed all three frozen order streams but was
owned-job-wall terminated before the required ID/reverse and future probes.
This module permits one new namespace for that exact model/seed.  It changes
only the owned wall resource bound; it does not reuse partial metrics, alter
the causal clock, or relax any data/GT/vocabulary boundary.
"""
import json
from pathlib import Path
from types import SimpleNamespace

from src.trackocd_v2.io import sha256_file
from src.trackocd_cvpr import M_member_input_development_profile as member_profile

CONFIG = 'configs/trackocd_cvpr/M_GT_causal_development_member_input_wall_recovery_v1.json'
FAILED = 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_member_input_memo_v1_relative_quarantine_s1027'
TERMINAL = FAILED + '_supervisor_attempt_1.json'
OLD_FILES = {
    FAILED + '/launch.json': '79b8b41b1b28c778a65c9319f466684d3f922e11ab83fd31a5a736f5ef237301',
    FAILED + '/progress.json': '8374eb69acac2d25ba02027bd3de12cd556a12ab32d71af8448229d41c295ed9',
    FAILED + '/owned_startup_observation_v1.json': 'f4aad36cc70d0a10029f59d3b8fd87e47b0918cea6291a130c2a204c646882fb',
    TERMINAL: '5fa18163351e49df16b5e04102ed82500c2396cb7618b62d68a5122ffb9c359f',
}
SOURCES = (CONFIG, 'src/trackocd_cvpr/M_member_input_wall_recovery.py',
           'scripts/trackocd_cvpr/evaluate_M_development_member_input_wall_recovery.py',
           *OLD_FILES)


def configuration(root, original, args):
    cfg = json.loads((root / CONFIG).read_text())
    expected_keys = {
        'status', 'base_config', 'base_config_sha256', 'original_protocol_config',
        'original_protocol_config_sha256', 'failed_namespace', 'failed_terminal',
        'failed_terminal_sha256', 'failed_launch_sha256', 'failed_progress_sha256',
        'failed_startup_sha256', 'model', 'seed', 'recovery_output_prefix',
        'original_wall_seconds_ceiling', 'recovery_wall_seconds_ceiling',
        'models_orders_prefixes_thresholds_history_and_metrics_unchanged',
        'only_wall_resource_bound_changed', 'partial_metric_reuse_or_old_namespace_overwrite',
        'optimizer_steps', 'training_GPU_hours', 'scientific_acceptance_or_model_selection',
        'recovery_is_not_original_3000_second_qualification'}
    if set(cfg) != expected_keys:
        raise ValueError('Exact wall-only recovery manifest required')
    if (cfg['status'] != 'REGISTERED_WALL_ONLY_RESOURCE_RECOVERY_MEMBER_INPUT_FULL_M'
            or cfg['base_config'] != member_profile.CONFIG
            or cfg['base_config_sha256'] != sha256_file(root / member_profile.CONFIG)
            or cfg['original_protocol_config_sha256'] != sha256_file(root / cfg['original_protocol_config'])
            or cfg['failed_namespace'] != FAILED or cfg['failed_terminal'] != TERMINAL
            or (args.model, args.seed) != (cfg['model'], cfg['seed'])
            or cfg['original_wall_seconds_ceiling'] != 3000
            or cfg['recovery_wall_seconds_ceiling'] != 6000
            or not cfg['models_orders_prefixes_thresholds_history_and_metrics_unchanged']
            or not cfg['only_wall_resource_bound_changed']
            or cfg['partial_metric_reuse_or_old_namespace_overwrite']
            or cfg['optimizer_steps'] != 0 or cfg['training_GPU_hours'] != 0
            or cfg['scientific_acceptance_or_model_selection']
            or not cfg['recovery_is_not_original_3000_second_qualification']):
        raise ValueError('Only fixed relative_quarantine/s1027 wall recovery is allowed')
    if original['wall_seconds_ceiling'] != cfg['original_wall_seconds_ceiling']:
        raise ValueError('Original 3000-second protocol must remain unchanged')
    for path, digest in OLD_FILES.items():
        p = root / path
        if p.is_symlink() or not p.is_file() or sha256_file(p) != digest:
            raise ValueError('Preserved failed attempt evidence changed: ' + path)
    terminal = json.loads((root / TERMINAL).read_text())
    if (terminal['guard_stop_reason'] != 'owned_job_wall'
            or terminal['owned_worker_exit_code'] != -9
            or terminal['complete_result_present']
            or terminal['successful_owned_execution']
            or terminal['training_GPU_hours'] != 0
            or terminal['peak_sampled_owned_worker_RSS_bytes'] >= 3 * 1024**3):
        raise ValueError('Recovery requires the recorded wall-only incomplete attempt')
    old_dir = root / FAILED
    if {p.name for p in old_dir.iterdir()} != {
            'launch.json', 'progress.json', 'owned_startup_observation_v1.json'}:
        raise ValueError('No result/profile/failure or unregistered partial metric may be reused')
    base = member_profile.configuration(root, original, args)
    out = dict(base)
    out['output_prefix'] = cfg['recovery_output_prefix']
    out['output_directory'] = cfg['recovery_output_prefix'] + '_' + args.model + '_s' + str(args.seed)
    out['wall_seconds_ceiling'] = cfg['recovery_wall_seconds_ceiling']
    return out


def recovery_evidence(root):
    cfg = json.loads((root / CONFIG).read_text())
    return {
        'status': cfg['status'],
        'failed_namespace': cfg['failed_namespace'],
        'failed_terminal_sha256': sha256_file(root / TERMINAL),
        'recovery_wall_seconds_ceiling': cfg['recovery_wall_seconds_ceiling'],
        'only_wall_resource_bound_changed': cfg['only_wall_resource_bound_changed'],
        'partial_metric_reuse_or_old_namespace_overwrite': cfg['partial_metric_reuse_or_old_namespace_overwrite'],
    }
