"""Fail-closed public receipt audit for the nine current-source M endpoints.

The audit reads only the explicitly listed public launch/progress/result/terminal
receipts.  It does not open private prediction histories or checkpoints and it
does not qualify M development, select a model, or read Val/Test payloads.
"""
import json
from pathlib import Path

from src.trackocd_v2.io import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONFIG = 'configs/trackocd_cvpr/M_current_source_endpoint_cohort_audit_v1.json'
AUDIT_OUTPUT = 'outputs/trackocd_cvpr/cvpr4/M_current_source_endpoint_cohort_audit_v1/result.json'


def _read(root, relative):
    return json.loads((root / relative).read_text())


def configuration(root=ROOT):
    manifest = _read(root, CONFIG)
    expected = {
        'status', 'endpoint_manifest', 'endpoint_manifest_sha256',
        'endpoint_source_manifest', 'endpoint_source_manifest_sha256',
        'output_prefix', 'cases', 'required_public_receipts',
        'fixed_optimizer_steps', 'execution_device', 'training_GPU_hours',
        'policy_clock', 'expected_result_status',
        'formal_M_development_or_model_selection',
        'model_selection_or_dev_final_selection',
        'no_private_data_weights_or_prediction_history_upload',
        'cohort_audit_is_not_M_qualification_or_goal_acceptance',
    }
    if set(manifest) != expected:
        raise ValueError('Exact current-source endpoint cohort manifest required')
    if (manifest['status'] != 'REGISTERED_CURRENT_SOURCE_ENDPOINT_COHORT_AUDIT_PENDING'
            or manifest['fixed_optimizer_steps'] != 200
            or manifest['execution_device'] != 'CPU'
            or manifest['training_GPU_hours'] != 0
            or manifest['policy_clock'] != {
                'videos': 10, 'physical_tracks': 54, 'observations': 1268,
                'metadata_frames': 374, 'empty_GT_frames': 1}
            or manifest['expected_result_status'] !=
                'COMPLETED_FIXED_M_TRIAL_NOT_DEV_PRED_JOINT_OR_GOAL_ACCEPTANCE'
            or manifest['formal_M_development_or_model_selection']
            or manifest['model_selection_or_dev_final_selection']
            or not manifest['no_private_data_weights_or_prediction_history_upload']
            or not manifest['cohort_audit_is_not_M_qualification_or_goal_acceptance']):
        raise ValueError('Cohort audit must remain public-receipt-only')
    endpoint_manifest_path = root / manifest['endpoint_manifest']
    source_manifest_path = root / manifest['endpoint_source_manifest']
    if (sha256_file(endpoint_manifest_path) != manifest['endpoint_manifest_sha256']
            or sha256_file(source_manifest_path) != manifest['endpoint_source_manifest_sha256']):
        raise ValueError('Registered endpoint manifest changed')
    endpoint = json.loads(endpoint_manifest_path.read_text())
    source = json.loads(source_manifest_path.read_text())
    if (endpoint['status'] != 'REGISTERED_CURRENT_SOURCE_ENDPOINT_TRANSPORT_RETRY1'
            or endpoint['recovery_output_prefix'] != manifest['output_prefix']
            or endpoint['base_manifest'] != manifest['endpoint_source_manifest']
            or endpoint['base_manifest_sha256'] != manifest['endpoint_source_manifest_sha256']
            or source['status'] != 'REGISTERED_CURRENT_SOURCE_FIXED_M_ENDPOINT_REFRESH_PENDING'
            or not isinstance(source.get('current_source_paths'), dict)
            or len(source['current_source_paths']) != 6):
        raise ValueError('Unexpected endpoint/source manifest')
    cases = manifest['cases']
    if len(cases) != 9 or len({(c['variant'], c['seed']) for c in cases}) != 9:
        raise ValueError('Exactly nine distinct endpoint cases are required')
    if any(not c['output_directory'].startswith(manifest['output_prefix'] + '_') for c in cases):
        raise ValueError('Endpoint namespace escaped registered prefix')
    return manifest, endpoint, source


def verify_public_case(root, case, manifest, source):
    namespace = case['output_directory']
    launch_path = namespace + '/launch.json'
    progress_path = namespace + '/progress.json'
    result_path = namespace + '/result.json'
    terminal_path = namespace + '_supervisor_attempt_1.json'
    for relative in (launch_path, progress_path, result_path, terminal_path):
        if not (root / relative).is_file():
            raise ValueError('Missing public endpoint receipt: ' + relative)
    launch = _read(root, launch_path)
    progress = _read(root, progress_path)
    result = _read(root, result_path)
    terminal = _read(root, terminal_path)
    proof = launch.get('live_remote_registration_proof')
    if not isinstance(proof, dict) or proof.get('method') != 'fresh_git_ls_remote':
        raise ValueError('Missing fresh remote registration proof: ' + namespace)
    if proof.get('ref') != 'refs/heads/research/cvpr-trackocd-autonomous-v2':
        raise ValueError('Unexpected registration ref: ' + namespace)
    if result.get('live_remote_registration_proof') != proof:
        raise ValueError('Launch/result registration proof mismatch: ' + namespace)
    current_paths = source['current_source_paths']
    source_hashes = launch.get('source_sha256', {})
    for path, digest in current_paths.items():
        if source_hashes.get(path) != digest:
            raise ValueError('Current source hash mismatch in ' + namespace + ': ' + path)
    if launch.get('scope') != 'Fixed200 fresh M trial/fourfullprediction-owned episodes':
        raise ValueError('Launch protocol flags failed: ' + namespace)
    if (progress.get('step') != manifest['fixed_optimizer_steps']
            or result.get('checkpoint_selection') != 'Fixed200 endpoint'
            or result.get('status') != manifest['expected_result_status']
            or result.get('summary', {}).get('optimizer_steps') != manifest['fixed_optimizer_steps']
            or result.get('endpoint', {}).get('reopen_exact') is not True
            or result.get('old_R_C_M_or_pilot_warmstart') is not False
            or result.get('dev_final_Val_Test_or_new_nonKnown_labels_used') is not False
            or result.get('resources', {}).get('CUDA_initialized') is not False
            or result.get('resources', {}).get('training_GPU_hours') != 0):
        raise ValueError('Endpoint result protocol flags failed: ' + namespace)
    if result.get('policy_scope') != {
            'GT_localization_oracle_control_not_qualified_Pred': True,
            'development_final_or_other_video_payload_bodies_opened': False,
            'development_final_or_other_video_semantic_values_decoded': False,
            'empty_GT_frames': 1,
            'existing_Known_target_counts': result['policy_scope']['existing_Known_target_counts'],
            'external_loss_only_not_observation_or_model_features': True,
            'masked_loss_tracks': result['policy_scope']['masked_loss_tracks'],
            'masked_tracks_dropped_or_declared_background': False,
            'metadata_frames': 374,
            'new_nonKnown_semantic_values_or_vocabulary_decoded': False,
            'observations': 1268,
            'physical_label_strata': result['policy_scope']['physical_label_strata'],
            'physical_tracks': 54,
            'videos': 10}:
        raise ValueError('Policy clock or semantic boundary failed: ' + namespace)
    if (terminal.get('owned_worker_exit_code') != 0
            or terminal.get('guard_stop_reason') is not None
            or terminal.get('complete_result_present') is not True
            or terminal.get('successful_owned_execution') is not True
            or terminal.get('last_persisted_completed_optimizer_steps') != manifest['fixed_optimizer_steps']
            or terminal.get('execution_device') != manifest['execution_device']
            or terminal.get('training_GPU_hours') != 0):
        raise ValueError('Terminal receipt failed: ' + namespace)
    return {
        'variant': case['variant'],
        'seed': case['seed'],
        'output_directory': namespace,
        'preregistration_commit': proof['sha'],
        'launch_sha256': sha256_file(root / launch_path),
        'progress_sha256': sha256_file(root / progress_path),
        'result_sha256': sha256_file(root / result_path),
        'terminal_sha256': sha256_file(root / terminal_path),
        'optimizer_steps': progress['step'],
        'terminal_wall_seconds': terminal['wall_seconds_including_imports_and_network'],
        'terminal_peak_RSS_bytes': terminal['peak_sampled_owned_worker_RSS_bytes'],
        'result_status': result['status'],
    }


def audit_cohort(root=ROOT):
    manifest, endpoint, source = configuration(root)
    receipts = [verify_public_case(root, case, manifest, source) for case in manifest['cases']]
    commits = {item['preregistration_commit'] for item in receipts}
    return {
        'status': 'COHORT_AUDIT_PASS_CURRENT_SOURCE_ENDPOINT_PROVENANCE_ONLY',
        'manifest': CONFIG,
        'manifest_sha256': sha256_file(root / CONFIG),
        'endpoint_manifest_sha256': manifest['endpoint_manifest_sha256'],
        'endpoint_source_manifest_sha256': manifest['endpoint_source_manifest_sha256'],
        'cases': receipts,
        'case_count': len(receipts),
        'distinct_preregistration_commits': sorted(commits),
        'all_public_receipts_normal': True,
        'all_current_source_hashes_match_registered_closure': True,
        'all_fixed_optimizer_steps': manifest['fixed_optimizer_steps'],
        'all_cpu_only': True,
        'training_GPU_hours': 0,
        'private_history_or_checkpoint_read': False,
        'private_asset_upload': False,
        'model_selection_or_dev_final_selection': False,
        'formal_M_development_or_model_selection': False,
        'M_qualification_or_goal_acceptance': False,
        'Val_Test_or_new_nonKnown_semantics_read': False,
    }
