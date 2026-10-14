"""Public verifier for the completed full-M source-compatibility replay.

The replay is a causal development diagnostic.  Its endpoint training
receipts remain historical, so this verifier deliberately does not promote
the result to current-source M qualification or Goal acceptance.
"""
import json
import re
from types import SimpleNamespace

from src.trackocd_v2.io import sha256_file
from scripts.trackocd_cvpr import evaluate_M_development as original
from .M_source_compatibility_M_recovery import configuration
from .M_source_compatibility_shadow_profile import verify_source_compatibility_shadow

REPLAY = (
    'outputs/trackocd_cvpr/cvpr4/'
    'M_GT_causal_dev_member_input_source_compatibility_v1_relative_quarantine_s1027'
)
TERMINAL = REPLAY + '_supervisor_attempt_1.json'
BINDING = REPLAY + '/source_compatibility_replay_binding.json'
PREREGISTRATION_COMMIT = '2ad0c69053f166790f1e6860cb42ea3e20f580a6'
MODEL = 'relative_quarantine'
SEED = 1027
STATUS = 'MEASURED_CAUSAL_GT_M_DEVELOPMENT_NOT_ACTIVE_PURITY_SELF_EXCLUSION_PRED_OR_GOAL_ACCEPTANCE'


def _regular_json(root, path):
    p = root / path
    if p.is_symlink() or not p.is_file():
        raise ValueError('Replay artifact must be a regular file: ' + path)
    return json.loads(p.read_text())


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None


def verify_source_compatibility_M_replay(root):
    args = SimpleNamespace(model=MODEL, seed=SEED)
    base = original.configuration(root, args)
    cfg = configuration(root, base, args)
    manifest = json.loads((root / 'configs/trackocd_cvpr/M_GT_causal_development_source_compatibility_v1.json').read_text())
    shadow = verify_source_compatibility_shadow(root)
    binding = _regular_json(root, BINDING)
    required = {
        'preregistration_commit', 'config_sha256', 'launch_sha256',
        'result_sha256', 'execution_profile_sha256', 'terminal_sha256',
        'compatibility_shadow_binding_sha256',
    }
    if set(binding) != required or not _hex(binding['preregistration_commit'], 40):
        raise ValueError('Complete source-compatibility replay binding required')
    if any(not _hex(binding[key], 64) for key in required if key != 'preregistration_commit'):
        raise ValueError('Malformed source-compatibility replay binding')
    if binding['preregistration_commit'] != PREREGISTRATION_COMMIT:
        raise ValueError('Unexpected replay preregistration commit')
    if binding['config_sha256'] != sha256_file(root / 'configs/trackocd_cvpr/M_GT_causal_development_source_compatibility_v1.json'):
        raise ValueError('Replay manifest changed after execution')
    if binding['compatibility_shadow_binding_sha256'] != manifest['compatibility_shadow_binding_sha256']:
        raise ValueError('Compatibility shadow binding changed')

    paths = {
        REPLAY + '/launch.json': binding['launch_sha256'],
        REPLAY + '/result.json': binding['result_sha256'],
        REPLAY + '/execution_profile.json': binding['execution_profile_sha256'],
        TERMINAL: binding['terminal_sha256'],
    }
    for path, digest in paths.items():
        if sha256_file(root / path) != digest:
            raise ValueError('Replay artifact hash mismatch: ' + path)

    launch = _regular_json(root, REPLAY + '/launch.json')
    result = _regular_json(root, REPLAY + '/result.json')
    profile = _regular_json(root, REPLAY + '/execution_profile.json')
    terminal = _regular_json(root, TERMINAL)
    proofs = (launch['live_remote_registration_proof'], result['live_remote_registration_proof'])
    if any(proof.get('sha') != PREREGISTRATION_COMMIT for proof in proofs):
        raise ValueError('Replay is not bound to its published preregistration')
    if launch['scope'] != 'ALL9 completed M endpoints/causal GT development/no selection':
        raise ValueError('Unexpected inherited worker launch scope')
    if (not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None or not terminal['complete_result_present']
            or terminal['training_GPU_hours'] != 0
            or terminal['peak_sampled_owned_worker_RSS_bytes'] > 3221225472
            or terminal['wall_seconds_including_imports_and_network'] > 6000):
        raise ValueError('Replay terminal is incomplete or outside its resource contract')
    if (result['status'] != STATUS or result['model'] != MODEL
            or result['training_seed'] != SEED or result['optimizer_steps'] != 0
            or result['model_or_checkpoint_selection']
            or result['final_Val_Test_or_new_nonKnown_semantic_values_read']
            or result['coverage_or_latency_matched_purity_superiority_accepted']
            or result['current_active_purity_official_CT_self_exclusion_Pred_joint_or_Goal_accepted']):
        raise ValueError('Replay result is not the fixed non-qualification diagnostic')
    resources = result['resources']
    if (resources['CUDA_initialized'] or resources['training_GPU_hours'] != 0
            or resources['peak_guarded_RSS_bytes'] > 3221225472):
        raise ValueError('Replay violated CPU/RSS/zero-GPU contract')
    if set(result['orders']) != {'1027', '1028', '1029'}:
        raise ValueError('All three fixed causal orders are required')
    for row in result['orders'].values():
        if (row['full_physical_bitset_scope'] != 472 or row['observations'] != 8753
                or row['metadata_frames'] != 2562 or row['empty_frames'] != 116
                or row['optimizer_steps'] != 0 or not row['model_state_unchanged']
                or not row['GT_oracle_control_not_Pred']):
            raise ValueError('Incomplete or mutable full-M order')
    scope = result['development_scope']
    if (scope['physical_tracks'] != 472 or scope['observations'] != 8753
            or scope['metadata_frames'] != 2562
            or scope['new_nonKnown_semantic_values_or_vocabulary_decoded']
            or scope['fit_policy_final_or_other_video_target_values_decoded']):
        raise ValueError('Replay scope or semantic boundary changed')
    if (not result['actual_full_ID_rename_reverse_stream_equal']
            or result['actual_future_probe']['future_observations_mutated'] <= 0
            or not result['actual_future_probe']['verified_same_prior_events_chosen_versions_and_numeric_ledger']):
        raise ValueError('Nonvacuous causal probes did not complete')
    if (profile['result_sha256'] != binding['result_sha256']
            or profile['execution_profile_sha256'] != manifest['compatibility_shadow_profile_sha256']
            or profile['terminal_sha256'] != manifest['compatibility_shadow_terminal_sha256']
            or not profile['actual_all11_current_source_deployment_trace_compatibility_checked']
            or profile['current_source_training_provenance_established']
            or not profile['compatibility_replay_is_not_formal_M_qualification']
            or profile['M_qualification_or_goal_acceptance']
            or profile['learned_outputs_or_partial_metrics_cached']
            or profile['Raw_statistics_bytes'] > 33554432
            or profile['fingerprint_bytes'] > 16252928
            or profile['member_input_bytes'] > 16777216
            or profile['temporary_index_peak_bookkeeping_bytes'] > 524288):
        raise ValueError('Replay profile is not a bounded compatibility diagnostic')
    return {
        **shadow,
        'actual_full_M_source_compatibility_replay_complete': True,
        'replay_result_sha256': binding['result_sha256'],
        'replay_terminal_sha256': binding['terminal_sha256'],
        'current_source_training_provenance_established': False,
        'M_qualification_or_goal_acceptance': False,
        'launch_scope_is_inherited_original_worker_label': True,
    }
