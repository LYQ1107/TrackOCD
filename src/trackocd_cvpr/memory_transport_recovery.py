"""One exact pre-input registration failure; namespace-only fresh M retry."""
import hashlib
import json
import subprocess
from src.trackocd_v2.io import sha256_file

CONFIG = 'configs/trackocd_cvpr/M_registration_transport_retry1.json'
FAILED_TERMINAL = 'outputs/trackocd_cvpr/cvpr4/M_GT_window_v1_pooled_weighted_s1028_supervisor_attempt_1.json'
RECOVERY_SOURCES = (CONFIG, 'src/trackocd_cvpr/memory_transport_recovery.py', FAILED_TERMINAL)
ORIGINAL = 'outputs/trackocd_cvpr/cvpr4/M_GT_window_v1_pooled_weighted_s1028'
RETRY = ORIGINAL + '_transport_retry1'


def verify_original_preinput_failure(root):
    r = json.loads((root / CONFIG).read_text())
    if (r['status'] != 'PRE_INPUT_TRANSPORT_RECOVERY_ONLY_SCIENTIFIC_WINDOW_UNCHANGED'
            or (r['allowed_variant'], r['allowed_seed']) != ('pooled_weighted', 1028)
            or r['original_attempt'] != ORIGINAL or r['output_directory'] != RETRY
            or r['original_terminal'] != FAILED_TERMINAL
            or r['original_execution_commit'] != 'de4dfdb92ee098ed049711668a7c3b17bbd6ebcd'
            or r['original_terminal_sha256'] != '311856168731045c5e6ac46ab4f5d5ac32fe0c59cb2268c59c3160f06ffb5afc'
            or r['original_worker_source_sha256'] != '6210712f0e2262117e62423fd463792f08ae3948a27bdaa16e0eb63e4acf0dca'
            or r['original_window'] != 'configs/trackocd_cvpr/M_GT_scientific_window_v1.json'
            or r['original_window_sha256'] != '626699f36bdf26e9cf464339a45a7a2a320b27fbe9fef597dd4f9f90eb779e3d'
            or r['resume_partial_metrics_or_training'] or r['original_attempt_overwrite']
            or r['change_orders_labels_thresholds_weights_initialization_steps_guards_or_selection']
            or r['original_tool_terminal_exit_code'] != 1 or (root / ORIGINAL).exists()):
        raise ValueError('Only exact recorded pre-input failure; no successful/partial trial retry')
    if any((root / path).is_symlink() for path in (CONFIG, FAILED_TERMINAL, r['original_window'])):
        raise ValueError('No redirected recovery evidence')
    if (sha256_file(root / FAILED_TERMINAL) != r['original_terminal_sha256']
            or sha256_file(root / r['original_window']) != r['original_window_sha256']):
        raise ValueError('Preserved failed terminal and unchanged scientific window required')
    t = json.loads((root / FAILED_TERMINAL).read_text())
    if (t['successful_owned_execution'] or t['owned_worker_exit_code'] != 1 or t['guard_stop_reason'] is not None
            or t['last_persisted_completed_optimizer_steps'] != 0 or t['training_GPU_hours'] != 0):
        raise ValueError('Actual normal terminal pre-input failure, not a scientific retry')
    source = subprocess.check_output(['git', 'show', r['original_execution_commit'] + ':scripts/trackocd_cvpr/pilot_memory_optimizer.py'], cwd=root)
    if hashlib.sha256(source).hexdigest() != r['original_worker_source_sha256']:
        raise ValueError('Original executed worker source binding required')
    return r


def recovery_configuration(root, cfg, attempt):
    if attempt is None: return cfg, ()
    if attempt == 'terminal_replica1':
        from .M_terminal_replica_recovery import recovery_configuration as terminal_replica
        return terminal_replica(root, cfg, attempt)
    r = verify_original_preinput_failure(root)
    if (attempt != 'transport_retry1' or not cfg.get('scientific_M_trial')
            or (cfg['variant'], cfg['seed']) != (r['allowed_variant'], r['allowed_seed'])
            or cfg['output_directory'] != ORIGINAL):
        raise ValueError('Only this exact fixed M trial namespace recovery')
    result = dict(cfg); result['output_directory'] = RETRY
    return result, RECOVERY_SOURCES


def cohort_directory(variant, seed):
    if (variant, seed) == ('pooled_weighted', 1028): return RETRY
    if (variant, seed) == ('relative_weighted', 1029):
        return f'outputs/trackocd_cvpr/cvpr4/M_GT_window_v1_{variant}_s{seed}_terminal_replica1'
    return f'outputs/trackocd_cvpr/cvpr4/M_GT_window_v1_{variant}_s{seed}'
