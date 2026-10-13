"""ALL4-only public comparison to exact Raw/ALL6 C, not model selection."""
import json
from .raw_membership_formal_cohort import TRIALS
from .raw_membership_development_receipts import verified_matrix, trial_sources
from .raw_membership_baseline_reference import verified_raw_reference, DIRECTORY
from .spread_C_development_complete_matrix import verified_matrix as verified_C, aggregate_matrix

PREFIXES = ('1', '2', '4', '8', '16', 'all_available')
ORDERS = ('1027', '1028', '1029')
FIELDS = ('Known_micro_accuracy', 'Known_macro_accuracy', 'all_physical_commit_track_coverage',
    'declared_pseudo_absorbed_as_Known_queries', 'declared_pseudo_anonymous_commit_queries',
    'certified_cross_video_reuse_micro_lower_bound', 'certified_cross_video_reuse_macro_lower_bound')


def order_means(result):
    if set(result['orders']) != set(ORDERS): raise ValueError('All3 actual orders, no best order')
    return {p:{k:sum(result['orders'][o]['prefix_snapshots'][p][k] for o in ORDERS)/3
               for k in FIELDS} for p in PREFIXES}


def complete_comparison(root):
    # Critical order: missing ALL4 normal endpoints refuses BEFORE historical
    # Raw/C aggregation. None of these gates opens private labels or weights.
    proof = verified_matrix(root); raw_proof = verified_raw_reference(root)
    c = verified_C(root)
    raw = json.loads((root/(DIRECTORY+'/result.json')).read_text())
    rows = {(control, seed):json.loads((root/trial_sources(control, seed)[1]).read_text())
            for control, seed in TRIALS}
    for r in (*rows.values(), *c.values()):
        if r['development_scope'] != raw['development_scope'] or r['offline_fit_prior_scope'] != raw['offline_fit_prior_scope']:
            raise ValueError('Same actual dev target/physical and complete offline fit prior scopes')
        for o in ORDERS:
            if r['orders'][o]['private_video_ordinal_order_sha256'] != raw['orders'][o]['private_video_ordinal_order_sha256']:
                raise ValueError('Same actual common full video ordinal schedule, not declaration-only')
    per_calendar = {str(seed):order_means(rows[('calendar_matched', seed)]) for seed in (1027, 1028, 1029)}
    mean_calendar = {p:{k:sum(per_calendar[str(seed)][p][k] for seed in (1027, 1028, 1029))/3
                       for k in FIELDS} for p in PREFIXES}
    raw_mean = order_means(raw); full = order_means(rows[('full_policy', None)])
    return {'status':'ALL4_ACTUAL_COMPLETE_PUBLIC_CONTROL_COMPARISON_NOT_DEPLOYMENT_CALIBRATION_JOINT_PRED_FINAL_OR_GOAL',
        'all4_proof':proof, 'historical_Raw_reference_proof':raw_proof,
        'fixed_Raw_zero_fit_order_mean':raw_mean,
        'calendar_matched':{'per_calendar_seed_order_mean':per_calendar,
            'three_calendar_seed_mean':mean_calendar,
            'three_calendar_seed_mean_minus_fixed_Raw':{p:{k:mean_calendar[p][k]-raw_mean[p][k] for k in FIELDS} for p in PREFIXES}},
        'ONE_deterministic_fullpolicy':{'order_mean':full,
            'minus_fixed_Raw':{p:{k:full[p][k]-raw_mean[p][k] for k in FIELDS} for p in PREFIXES},
            'independent_training_seed_runs':False},
        'ALL6_previous_paired_C_comparison':aggregate_matrix(c),
        'independent_conditional_reliability_per_endpoint':[
            {'control':control, 'calendar_seed':seed, 'diagnostics':rows[(control, seed)]['conditional_reliability']}
            for control, seed in TRIALS],
        'calendar_matched_to_C_means_same_actual_frame_updates_not_capacity_targets_or_gradient_exposure':True,
        'fullpolicy_is_calendar_or_optimizer_matched_to_C':False,
        'class_macro_or_role_conditioning_is_deployment_prior':False,
        'order_replays_prefixes_observations_are_independent_replications':False,
        'control_seed_order_prefix_bin_threshold_or_checkpoint_selected':False,
        'fair_actual_latency_coverage_matched_foreground_Existing_purity_fullposterior_claimed':False,
        'full_semantic_GT_Hscore_CT_R_M_Pred_dense_lockedfinal_or_Goal_accepted':False,
        'private_labels_weights_scalar_target_tables_or_prediction_rows_exported':False}
