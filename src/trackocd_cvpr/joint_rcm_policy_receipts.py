"""Public checks for the bounded joint R/C/M pilot receipt.

Only aggregate result and supervisor receipts are read.  Private checkpoints,
loss curves, and per-object prediction rows are deliberately not required.
"""

import json
import re
from pathlib import Path


def verify_joint_rcm_policy_receipt(root):
    root = Path(root)
    directory = root / 'outputs/trackocd_cvpr/cvpr5/joint_rcm_policy_pilot_v1_repair2'
    result = json.loads((directory / 'result.json').read_text())
    terminal = json.loads((directory.parent / 'joint_rcm_policy_pilot_v1_repair2_supervisor_attempt_1.json').read_text())
    proof = result['live_remote_registration_proof']
    if (result['status'] != 'COMPLETED_JOINT_RCM_MEMORY_PILOT_NOT_DEV_PRED_OR_GOAL_ACCEPTANCE'
            or proof['ref'] != 'refs/heads/research/cvpr-trackocd-autonomous-v2'
            or not re.fullmatch(r'[0-9a-f]{40}', proof['sha'])
            or not terminal['successful_owned_execution']
            or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None
            or result['model_selection_or_goal_acceptance']
            or result['dev_final_Val_Test_or_new_nonKnown_semantics_used']
            or result['private_endpoint']['content_uploaded']
            or not result['private_endpoint']['reopened_exact']
            or result['resources']['training_GPU_hours'] != 0
            or result['resources']['CUDA_initialized']):
        raise ValueError('Joint R/C/M pilot receipt safety/status contract failed')
    scope = result['pilot_scope']; summary = result['summary']
    episode = summary['episodes'][0]
    lineage = episode['version_lineage_audit']
    curve = summary['loss_curve']
    if (scope != {'policy_relative_ordinals': [0, 1], 'videos': 2,
                  'selection_frozen_before_payload_read': True,
                  'not_development_or_final': True}
            or summary['optimizer_steps'] != 2
            or summary['trainable_parameters'] != 51078
            or not summary['joint_graph_reached_R_C_and_M']
            or not summary['prediction_owned_memory_only']
            or summary['GT_or_id_or_token_in_numeric_forward']
            or summary['dev_final_Val_Test_or_new_nonKnown_semantics_read']
            or summary['development_or_goal_acceptance']
            or episode['metadata_frames'] != 79
            or episode['empty_frames'] != 1
            or episode['physical_observations'] != 215
            or episode['canonical_groups'] != 215
            or episode['prediction_owned_commit_records'] != 75
            or episode['version_records'] != 143
            or lineage['actual_commits'] != 75
            or not lineage['all_parent_history_and_empty_tombstones_retained']
            or len(curve) != 2
            or any(min(row[name] for name in ('R_gradient_norm_sum',
                                               'C_gradient_norm_sum',
                                               'M_gradient_norm_sum')) <= 0 for row in curve)):
        raise ValueError('Joint R/C/M pilot aggregate contract failed')
    policy = result['policy_scope']; fit = result['offline_fit_prior_scope']
    if (policy['development_final_or_other_video_payload_bodies_opened']
            or not policy['GT_localization_oracle_control_not_qualified_Pred']
            or policy['new_nonKnown_semantic_values_or_vocabulary_decoded']
            or not policy['external_loss_only_not_observation_or_model_features']
            or not fit['targets_external_to_numeric_forward']
            or fit['unlabeled_or_other_Known_assigned_background_or_Novel']):
        raise ValueError('Joint R/C/M data-role boundary failed')
    return {'status': result['status'], 'remote_ref': proof['sha'],
            'optimizer_steps': summary['optimizer_steps'], 'videos': scope['videos'],
            'metadata_frames': episode['metadata_frames'],
            'physical_observations': episode['physical_observations'],
            'version_records': episode['version_records']}
