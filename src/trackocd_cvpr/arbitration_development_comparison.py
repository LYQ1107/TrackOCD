"""Complete public all-five observer evidence, never a rollout intervention gain."""
import json
from math import fsum
from . import arbitration_development_receipts as receipts
from .arbitration_development_protocol import EXPECTED


def complete_comparison(root):
    # Refuse incomplete/changed/unsafe trials before loading any comparison inputs.
    proof = receipts.verified_matrix(root)
    models = tuple(EXPECTED['models'])
    rows = {}
    metrics = {
        'Known_Existing_count':('Known','EXISTING'),
        'Known_only_own_support_count':('Known','Existing_only_own_groups'),
        'Known_positive_gate_bypassed_count':('Known','positive_named_gate_bypassed_groups'),
        'Known_correct_one_step_shadow_count':('Known','same_history_remove_own_correct_Known_groups'),
        'declared_pseudo_Known_one_step_shadow_count':('policy_pseudo','same_history_remove_own_KNOWN_groups'),
        'masked_Known_one_step_shadow_count_no_correctness_truth':('masked','same_history_remove_own_KNOWN_groups'),
    }
    for model in models:
        paths = receipts.trial_sources(model)
        r = json.loads((root/paths[1]).read_text())
        t = json.loads((root/paths[2]).read_text())
        rows[model] = {
            'original_public_triplet':list(paths),
            'full_orders':{o:{'full_clock':v['measured']['full_clock'],
                'external_strata':v['measured']['external_strata']} for o,v in r['orders'].items()},
            'mean_counts_over3orders_not_independent_trials':{
                key:fsum(r['orders'][o]['measured']['external_strata'][s].get(k,0)
                    for o in ('1027','1028','1029'))/3 for key,(s,k) in metrics.items()},
            'normal_wall_seconds':t['wall_seconds_including_worker_imports_network'],
            'normal_peak_RSS_bytes':t['peak_sampled_owned_worker_RSS_bytes'],
        }
    calendar = ('calendar1027','calendar1028','calendar1029')
    means = {key:fsum(rows[m]['mean_counts_over3orders_not_independent_trials'][key]
        for m in calendar)/3 for key in metrics}
    return {
        'status':'ALL5_ACTUAL_COMPLETE_PUBLIC_TRAIN_DEV_OBSERVER_COMPARISON_NOT_INTERVENED_STREAM_OR_GOAL',
        'qualification':proof,'models':rows,
        'three_calendar_seed_mean_over_all3orders':means,
        'per_order_observation_denominators':{'Known':4808,'declared_pseudoNovel':2743,'masked':1202},
        'normal_total_wall_seconds':fsum(rows[m]['normal_wall_seconds'] for m in models),
        'training_GPU_hours':0,'private_GT_weights_rows_loaded':False,
        'conditional_shadow_is_full_intervention_gain':False,
        'observations_orders_or_solver_replicated_as_independent_seeds':False,
        'model_seed_threshold_or_coefficient_selected':False,
        'joint_Novel_R_C_M_Pred_dense_calibrated_controls_final_or_Goal_accepted':False,
    }
