"""Frozen public prerequisites for a read-only policy attribution pilot."""
import json
from src.trackocd_v2.io import sha256_file
from .raw_membership_control_comparison import complete_comparison

CONFIG = 'configs/trackocd_cvpr/arbitration_policy_attribution_pilot_v1.json'
REPORT = 'outputs/trackocd_cvpr/cvpr3/raw_NamedKnown_membership_all4_full_dev_comparison_v1.json'
REPORT_SHA = 'dbda23e50e41ee9d0d2fb0a5cc49bd3af0d42fab5bd106d75e79057ea3f8a072'
OUT = 'outputs/trackocd_cvpr/cvpr3/arbitration_policy_attribution_pilot_v1'
EXPECTED = {
    'status':'REGISTERED_READONLY_POLICY_ARBITRATION_ATTRIBUTION_PILOT_ALL4_DEV_COMPLETE_REQUIRED',
    'base_data_config':'configs/trackocd_cvpr/candidate_optimizer_pilot_v1.json',
    'all4_dev_comparison':REPORT,'all4_dev_comparison_sha256':REPORT_SHA,
    'models':['initial_Raw','calendar1027','calendar1028','calendar1029','ONE_fullpolicy'],
    'order_seed':1027,'policy_videos':10,'physical_tracks':54,'observations':1268,
    'metadata_frames':374,'empty_frames':1,'Known_tracks':34,'policy_pseudo_tracks':10,'masked_tracks':10,
    'K':8,'optimizer_steps':0,'execution_device':'CPU','CPU_compute_workers':1,
    'rss_bytes_ceiling':3221225472,'wall_seconds_ceiling':3000,'output_bytes_ceiling':16777216,
    'output_directory':OUT,'maximum_actual_pilot_attempts':1,
    'all13_private_snapshot_fingerprints_before_policy_input':True,
    'full_normal_reference_event_commit_state_parity_per_model':True,
    'full_ID_reverse_and_nonvacuous_future_prefix_probes_per_model':True,
    'shadow_exclusion_writes_or_retroactive_GT_history_edit':False,
    'conditional_one_decision_shadow_is_full_intervention_rollout_gain':False,
    'labels_provided_to_scorer_observer_or_support_exclusion':False,
    'unknown_as_negative_BG_Novel_or_semantic_vocabulary':False,
    'dev_final_Val_Test_original_pixels_or_new_nonKnown_GT_read':False,
    'optimizer_model_seed_threshold_selection_or_auto_formal_expansion':False,
    'policy_resubstitution_is_unseen_dev_causal_mechanism_or_Goal_acceptance':False,
}


def configuration(root):
    cfg = json.loads((root/CONFIG).read_text())
    if cfg != EXPECTED:raise ValueError('Exact finite all5 policy attribution pilot, not selected subset/intervention/role expansion')
    if (root/REPORT).is_symlink() or sha256_file(root/REPORT) != REPORT_SHA:
        raise ValueError('Exact actual complete ALL4 public comparison first')
    public = json.loads((root/REPORT).read_text()); actual = complete_comparison(root)
    if any(public[k] != v for k,v in actual.items()):raise ValueError('All original actual complete matrix prerequisite')
    base = json.loads((root/cfg['base_data_config']).read_text())
    # Reuse DATA lineage only, never inherit the old optimizer/model/schedule
    # descriptions into a read-only observer's runtime configuration.
    data_keys = ('approval','frozen_protocol','private_assignment','private_Known_support',
        'fit_cache_directory','fit_cache_execution_commit','policy_dev_cache_directory',
        'policy_dev_execution_commit','policy_dev_result_sha256','policy_input_qualification_sha256')
    return {**{k:base[k] for k in data_keys},**cfg}
