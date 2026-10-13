"""One exact source-bound pre-input C failure; full namespace-only replay."""
import hashlib
import json
import subprocess
from src.trackocd_v2.io import sha256_file

CONFIG = 'configs/trackocd_cvpr/C_registration_transport_retry1.json'
ORIGINAL = 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Top8_pool_context_s1028'
TERMINAL = ORIGINAL + '_supervisor_attempt_1.json'
PREFIX = 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_registration_retry1'


def registration_recovery(root, cfg, args):
    r = json.loads((root / CONFIG).read_text())
    if (args.recovery_attempt != 'registration_retry1' or args.model != 'Top8_pool_context' or args.seed != 1028
            or r['status'] != 'PRE_INPUT_REGISTRATION_RECOVERY_ONLY_C_PROTOCOL_UNCHANGED'
            or (r['allowed_model'], r['allowed_seed']) != (args.model, args.seed)
            or r['original_attempt'] != ORIGINAL or r['original_terminal'] != TERMINAL
            or r['output_prefix'] != PREFIX
            or r['original_config'] != 'configs/trackocd_cvpr/C_GT_causal_development_v1.json'
            or r['original_config_sha256'] != '14b408e643958f2ce05f55462bbde5ac673eab319ec9e957e55da8a95e951657'
            or r['original_execution_commit'] != '21f3d6180cdbcad9165cfacf77e2cca8d72de14c'
            or r['original_worker_source_sha256'] != 'b8a109eee2eeb889d3172ee26100e712565b0eb5d239bf313753f109b1927f22'
            or r['original_terminal_sha256'] != 'e6315ce9eb14c5b583c8ecfe9d3e46a6a7eb8abb13e72e342e1c78a9292a987c'
            or r['local_only_original_stdout'] != ORIGINAL + '_persistent_stdout.log'
            or r['local_only_original_stdout_sha256'] != 'fcde981a9318614581507b66431bc88a62986976b027a6dfcc8548fe1597edd3'
            or r['resume_partial_metrics_or_change_scientific_protocol_or_guards'] or r['original_attempt_overwrite']
            or (root / ORIGINAL).exists()):
        raise ValueError('Exact C pre-input registration failure only, not a selected scientific rerun')
    for p, expected in ((r['original_config'], r['original_config_sha256']), (TERMINAL, r['original_terminal_sha256']),
                        (r['local_only_original_stdout'], r['local_only_original_stdout_sha256'])):
        if (root / p).is_symlink() or sha256_file(root / p) != expected:
            raise ValueError('Immutable original source/config/failed terminal/local traceback required')
    source = subprocess.check_output(['git', 'show', r['original_execution_commit'] + ':scripts/trackocd_cvpr/evaluate_C_development.py'], cwd=root)
    if hashlib.sha256(source).hexdigest() != r['original_worker_source_sha256']:
        raise ValueError('Actual original C worker source binding')
    t = json.loads((root / TERMINAL).read_text())
    if t['successful_owned_execution'] or t['owned_worker_exit_code'] != 1 or t['guard_stop_reason'] is not None or t['training_GPU_hours'] != 0:
        raise ValueError('Actual terminal pre-input failure; no guard/partial-result rescue')
    expected_original = cfg['output_prefix'] + '_' + args.model + '_s' + str(args.seed)
    if expected_original != ORIGINAL:
        raise ValueError('Only original fixed C development namespace')
    result = dict(cfg); result['output_prefix'] = PREFIX
    return result, (CONFIG, 'src/trackocd_cvpr/C_registration_recovery.py', TERMINAL)
