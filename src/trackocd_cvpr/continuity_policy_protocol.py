"""Exact prospective whole-policy intervention window; public gates first."""
import json
from src.trackocd_v2.io import sha256_file
from .arbitration_attribution_protocol import configuration as prior_policy_configuration
from .arbitration_development_comparison import complete_comparison
from .continuity_reuse_control import MODES

CONFIG='configs/trackocd_cvpr/continuity_reuse_policy_pilot_v1.json'
SOURCE_CONFIG='configs/trackocd_cvpr/source_continuity_reuse_factorization_v1.json'
SOURCE_SHA='5e80e98c2b0826b3571d9196dafb71b7563e4cfd9443f19a872f1ac06780725f'
REPORT='outputs/trackocd_cvpr/cvpr3/arbitration_all5_full_development_comparison_v1.json'
REPORT_SHA='45fdb286b5cf94a857fa3f1beb7d15ae4174f8698dba6e125113debb6c003c8d'
PREFIX='outputs/trackocd_cvpr/cvpr3/continuity_reuse_policy_pilot_v1'
EXPECTED={
    'status':'REGISTERED_ALL5_HEADS_FOUR_ARM_COMPLETE_POLICY_INTERVENTION_PILOT_NOT_DEV_TRAINING_OR_GOAL',
    'models':['initial_Raw','calendar1027','calendar1028','calendar1029','ONE_fullpolicy'],
    'arms':list(MODES),'order_seeds':[1027,1028,1029],'prefix_snapshots':[1,2,4,8,16],
    'policy_videos':10,'physical_tracks':54,'observations':1268,'metadata_frames':374,'empty_frames':1,
    'Known_tracks':34,'policy_pseudo_tracks':10,'masked_tracks':10,'K':8,'optimizer_steps':0,
    'execution_device':'CPU','CPU_compute_workers':1,'rss_bytes_ceiling':3221225472,
    'wall_seconds_ceiling':3000,'output_bytes_ceiling':16777216,'output_prefix':PREFIX,
    'maximum_actual_pilot_attempts_per_model':1,'one_explicit_model_per_worker':True,
    'all13_private_snapshot_fingerprints_before_policy_input':True,
    'full_original_event_commit_state_parity_for_EACH_baseline_order':True,
    'full_ID_reverse_nonvacuous_future_and_external_label_swap_per_arm':True,
    'full_known_output_binding_shuffle_per_arm':True,
    'actual_closed_loop_prediction_owned_commits_not_same_history_shadows':True,
    'origin_aware_primary_reuse_excludes_routing_continuation':True,
    'complete_physical_masked_denominators_and_unresolved_latency':True,
    'conservative_full_dirty_history_and_original_legacy_audit_retained':True,
    'GT_origin_or_future_truth_to_policy_or_memory_repair':False,
    'unknown_as_background_negative_or_new_Novel_semantics':False,
    'dev_final_Val_Test_pixels_or_new_nonKnown_GT_read':False,
    'head_seed_threshold_selection_or_automatic_training':False,
    'policy_resubstitution_is_unseen_dev_Pred_dense_joint_or_Goal_gain':False,
}


def configuration(root,model):
    cfg=json.loads((root/CONFIG).read_text())
    if cfg!=EXPECTED or model not in EXPECTED['models']:
        raise ValueError('Exact all5/four-arm/full3orders/policy clock/guards, no selected or partial pilot')
    for p,h in ((SOURCE_CONFIG,SOURCE_SHA),(REPORT,REPORT_SHA)):
        if (root/p).is_symlink() or sha256_file(root/p)!=h:raise ValueError('Exact qualified source and complete all5 prerequisite')
    if json.loads((root/REPORT).read_text())!=complete_comparison(root):raise ValueError('Actual full all5 prerequisite, not source claims')
    prior=prior_policy_configuration(root)
    keys=('approval','frozen_protocol','private_assignment','private_Known_support','fit_cache_directory',
          'fit_cache_execution_commit','policy_dev_cache_directory','policy_dev_execution_commit',
          'policy_dev_result_sha256','policy_input_qualification_sha256')
    return {**{k:prior[k] for k in keys},**cfg,'model':model,'output_directory':PREFIX+'_'+model}
