"""Public-only source-window/prerequisite validation; grants no asset access."""
import json
from src.trackocd_v2.io import sha256_file
from .C_development_complete_matrix import verified_matrix

CONFIG = 'configs/trackocd_cvpr/C_paired_known_set_rejection_v1.json'


def source_window(root):
    cfg = json.loads((root / CONFIG).read_text())
    expected = {
        'status':'SOURCE_ONLY_FROZEN_PAIRED_C_REJECTION_WINDOW_ACTUAL_PILOT_REQUIRED',
        'implementation':'src/trackocd_cvpr/paired_known_rejection_training.py',
        'authority':'configs/trackocd_cvpr/formal_research_scope_approval_v1.json',
        'data_protocol':'configs/trackocd_cvpr/frozen_global_data_GT_control_v1.json',
        'arms':['baseline','Known_set_rejection'],'trainable_parameters_each':708,
        'training_seeds':[1027,1028,1029],'repair_coefficient':.25,
        'data_scope':{'policy_videos':10,'physical_tracks':54,'observations':1268,'metadata_frames':374,
            'empty_frames':1,'Known_targets':34,'policy_pseudo_targets':10,'masked_unlabeled_tracks':10},
        'training':{'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'gradient_clip':1.,
            'episodes':4,'updates_per_episode_each':50,'optimizer_steps_each':200,
            'checkpoint_steps':[20,50,100,200],'endpoint':'Fixed200, no dev/final/Val/Test selection'},
        'pilot':{'seed':1027,'episodes':1,'updates_per_episode_each':50,'optimizer_steps_each':50,
            'checkpoint_steps':[50],'full_policy_clock_retained':True,
            'require_actual_nonzero_rejection_loss_gradient_before_expansion':True,'warmstart_formal_from_pilot':False},
        'formal_expansion_before_successful_pilot':False,'maximum_evidence_driven_repair_windows':1,
        'calibrated_Raw_thresholds_or_posterior_claimed':False,
        'Raw_calibration_and_coverage_latency_matched_controls_still_required':True,
        'new_nonKnown_GT_unlabeled_as_BG_or_Novel_supervision':False,
        'dev_final_Val_Test_or_original_pixels_in_training':False,
        'execution_device':'CPU','CPU_compute_workers':1,'rss_bytes_ceiling':3221225472,
        'wall_seconds_ceiling':3000,'output_bytes_ceiling':16777216,
        'private_input_reads_training_development_or_Goal_acceptance_from_source':False,'automatic_training_launch':False,
    }
    if any(cfg.get(key) != value for key,value in expected.items()):
        raise ValueError('Frozen paired source window, no coefficient/seed/denominator/guard/selection expansion')
    paths = {
        'original_complete_C_matrix_required':('outputs/trackocd_cvpr/cvpr3/C_development_all9_matrix_v1/result.json',
            '45911586efe6112045c4233aa47a685764b2e166f286122398b29baaf9ba0e3d'),
        'original_historical_loss_audit_required':('outputs/trackocd_cvpr/cvpr3/C_historical_loss_plan_audit_v1/result.json',
            '53522917edfe26ec097a511bd88ab28d65db751d034be2b0f9726a9c2ca6b9a3'),
    }
    for field,(path,digest) in paths.items():
        hash_field = field.replace('_required','_sha256')
        if cfg[field] != path or cfg[hash_field] != digest or (root/path).is_symlink() or sha256_file(root/path) != digest:
            raise ValueError('Actual full old C matrix/loss reconstruction prerequisite, not chat/source claim')
    if len(verified_matrix(root)) != 9:
        raise ValueError('Complete actual original matrix with preserved completion qualifications')
    audit_path = root / cfg['original_historical_loss_audit_required']
    audit = json.loads(audit_path.read_text())
    terminal = json.loads(audit_path.parent.with_name(audit_path.parent.name+'_supervisor_attempt_1.json').read_text())
    if (audit['status'] != 'ACTUAL_ALL9_C_HISTORY_LOSS_PLAN_RECONSTRUCTION_NOT_LOGITS_GRADIENTS_CAUSE_OR_GOAL'
            or audit['new_optimizer_steps'] != 0 or audit['new_inference_collapse_cause_development_Pred_or_Goal_accepted']
            or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None):
        raise ValueError('Genuine normal complete read-only audit, not invented missing NEW gradients')
    return cfg
