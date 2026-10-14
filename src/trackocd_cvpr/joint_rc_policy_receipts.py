"""Public checks for the bounded joint R/C pilot receipt.

The verifier reads only the aggregate result and owned terminal receipt. The
private checkpoint and prediction-owned rows are intentionally not required.
"""

import json
import re
from pathlib import Path


def verify_joint_rc_policy_receipt(root):
    root = Path(root)
    directory = root / 'outputs/trackocd_cvpr/cvpr5/joint_rc_policy_pilot_v1'
    result = json.loads((directory / 'result.json').read_text())
    terminal = json.loads((directory.parent / 'joint_rc_policy_pilot_v1_supervisor_attempt_1.json').read_text())
    proof = result['live_remote_registration_proof']
    if (result['status'] != 'COMPLETED_JOINT_RC_MEMORY_TRAIN_PILOT_NOT_DEV_PRED_OR_GOAL_ACCEPTANCE'
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
        raise ValueError('Joint pilot receipt safety/status contract failed')
    scope = result['pilot_scope']; summary = result['summary']
    if (scope != {'not_development_or_final': True, 'policy_relative_ordinals': [0, 1],
                  'selection_frozen_before_payload_read': True, 'videos': 2}
            or summary['optimizer_steps'] != 2
            or not summary['joint_graph_reached_R_and_C']
            or not summary['prediction_owned_memory_only']
            or summary['GT_or_id_or_token_in_numeric_forward']
            or not summary['all_selected_pilot_frames_and_observations_consumed']
            or len(summary['episodes']) != 1
            or summary['episodes'][0]['metadata_frames'] != 79
            or summary['episodes'][0]['physical_observations'] != 215
            or summary['episodes'][0]['canonical_groups'] != 215):
        raise ValueError('Joint pilot aggregate contract failed')
    return {'status': result['status'], 'remote_ref': proof['sha'],
            'optimizer_steps': summary['optimizer_steps'], 'videos': scope['videos'],
            'metadata_frames': summary['episodes'][0]['metadata_frames'],
            'physical_observations': summary['episodes'][0]['physical_observations']}
