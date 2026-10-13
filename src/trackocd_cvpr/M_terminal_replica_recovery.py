"""Exact completion replication for an unobserved owned M terminal, not tuning."""
import hashlib
import json
import subprocess
from src.trackocd_v2.io import sha256_file

CONFIG = 'configs/trackocd_cvpr/M_terminal_replica1.json'
ORIGINAL = 'outputs/trackocd_cvpr/cvpr4/M_GT_window_v1_relative_weighted_s1029'
REPLICA = ORIGINAL + '_terminal_replica1'
OBSERVATION = ORIGINAL + '_terminal_observation.json'
SOURCES = (CONFIG, 'src/trackocd_cvpr/M_terminal_replica_recovery.py', OBSERVATION,
           ORIGINAL + '/result.json', ORIGINAL + '/launch.json')


def verify_original(root, *, check_private=False):
    cfg = json.loads((root / CONFIG).read_text()); note = json.loads((root / OBSERVATION).read_text())
    if (cfg['status'] != 'EXACT_NUMERIC_COMPLETION_REPLICA_NOT_METRIC_SELECTION_OR_SCIENTIFIC_REPAIR'
            or (cfg['allowed_variant'], cfg['allowed_seed']) != ('relative_weighted', 1029)
            or cfg['original_attempt'] != ORIGINAL or cfg['output_directory'] != REPLICA or cfg['observation'] != OBSERVATION
            or cfg['original_window_sha256'] != '626699f36bdf26e9cf464339a45a7a2a320b27fbe9fef597dd4f9f90eb779e3d'
            or cfg['original_normal_exit_claim'] or cfg['overwrite_original_or_resume_partial_training']
            or cfg['change_seed_orders_labels_weights_steps_losses_or_guards']
            or note['status'] != 'COMPLETE_WORKER_ARTIFACTS_WITH_UNOBSERVED_OWNED_EXIT_NOT_NORMAL_TERMINAL_QUALIFIED'
            or note['reported_tool_exit_code'] != 143 or note['actual_owned_worker_wait_exit_code_observed']
            or note['normal_supervisor_terminal_present_at_observation'] or note['normal_owned_completion_or_scientific_win_accepted']
            or note['original_execution_commit'] != '91f9ba8ca1966cf7380c9e9ee5812c71bd6b737e'
            or note['original_worker_source_sha256'] != '0cd6cd3b5fda786ff49c24c927444511ce1bf40511b42da16613b09e862399bf'
            or note['original_launch_sha256'] != '2e60366cd16913c0935c110ddc0ce07bcffce32b4f054e06b4b82b0ed925a238'
            or note['original_result_sha256'] != '45bcbebcb47c3a66707c6da12851c0bd3afb0ea7ed6c196a52e20b53a472ada1'
            or (root / (ORIGINAL + '_supervisor_attempt_1.json')).exists()):
        raise ValueError('One exact unobserved-terminal fit, never a normal scientific result rerun')
    for p, expected in ((ORIGINAL + '/result.json', note['original_result_sha256']),
                        (ORIGINAL + '/launch.json', note['original_launch_sha256']),
                        ('configs/trackocd_cvpr/M_GT_scientific_window_v1.json', cfg['original_window_sha256'])):
        if (root / p).is_symlink() or sha256_file(root / p) != expected:
            raise ValueError('Immutable original completed artifacts/window required')
    source = subprocess.check_output(['git', 'show', note['original_execution_commit'] + ':scripts/trackocd_cvpr/pilot_memory_optimizer.py'], cwd=root)
    if hashlib.sha256(source).hexdigest() != note['original_worker_source_sha256']:
        raise ValueError('Actual original executed wrapper source binding')
    r = json.loads((root / (ORIGINAL + '/result.json')).read_text())
    if (r['seed'] != 1029 or r['variant'] != 'relative_weighted' or r['summary']['optimizer_steps'] != 200
            or r['checkpoint_selection'] != 'Fixed200 endpoint' or not r['endpoint']['reopen_exact']
            or r['old_R_C_M_or_pilot_warmstart'] or r['dev_final_Val_Test_or_new_nonKnown_labels_used']):
        raise ValueError('Exact existing full fixed trial, not partial continuation')
    if check_private:
        # Caller invokes this ONLY after fresh registration and authority gate.
        for p, record in ((root / ORIGINAL / 'private_step200_checkpoint.pt', r['endpoint']),
                          (root / ORIGINAL / 'private_prediction_history.json', r['private_prediction_history'])):
            if p.is_symlink() or p.stat().st_size != record['bytes'] or sha256_file(p) != record['sha256']:
                raise ValueError('Actual original full endpoint/history fingerprint')
    return r


def recovery_configuration(root, cfg, attempt):
    if attempt != 'terminal_replica1' or not cfg.get('scientific_M_trial') or (cfg['variant'], cfg['seed']) != ('relative_weighted', 1029) or cfg['output_directory'] != ORIGINAL:
        raise ValueError('Exact designated full terminal replica only')
    verify_original(root)
    out = dict(cfg); out['output_directory'] = REPLICA
    return out, SOURCES


def verify_replica_result(root, result):
    old = verify_original(root)
    for field in ('status', 'variant', 'seed', 'summary', 'policy_scope', 'offline_fit_prior_scope',
                  'endpoint', 'private_prediction_history', 'checkpoint_selection',
                  'old_R_C_M_or_pilot_warmstart', 'dev_final_Val_Test_or_new_nonKnown_labels_used'):
        if result[field] != old[field]:
            raise ValueError('Terminal replica numeric/optimizer/RNG/history/science difference: ' + field)
    return {'entire_fixed_summary_endpoint_optimizer_RNG_and_private_history_exact': True,
            'original_owned_wait_exit_observed': False, 'original_artifacts_overwritten': False,
            'no_metric_selection_or_new_scientific_seed': True}
