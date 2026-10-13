"""Posthoc public-result aggregation, no selector, data or checkpoint reads."""
import statistics
from .fresh_A_baseline import FAMILIES

CAPS = ('1', '2', '4', '8', '16', 'all_available')
SEEDS = ('1027', '1028', '1029')


def measurements(case):
    retrieval, known = case['retrieval'], case['Known']
    return {
        'R1_macro': retrieval['certified_lower_bound_macro_R1'],
        'R5_macro': retrieval['certified_lower_bound_macro_R5'],
        'R1_micro': retrieval['certified_lower_bound_micro_R1'],
        'R5_micro': retrieval['certified_lower_bound_micro_R5'],
        'Known_micro': known['micro_accuracy_exact_tie_expectation'],
        'Known_macro': known['macro_accuracy_exact_tie_expectation'],
        'effective_rank': case['effective_rank_full_physical'],
    }


def statistics_by_seed(values):
    if set(values) != set(SEEDS):
        raise ValueError('All three actual training seeds, no subset')
    rows = [values[s] for s in SEEDS]
    return {'mean': statistics.mean(rows), 'sample_std': statistics.stdev(rows),
            'min': min(rows), 'max': max(rows), 'per_seed': dict(values)}


def summarize(result, terminal):
    if (result['status'] != 'MEASURED_ALL15_FRESH_GT_BASELINE_DEV_NOT_JOINT_PRED_FINAL_OR_GOAL_ACCEPTANCE'
            or terminal['owned_worker_exit_code'] != 0 or terminal['guard_stop_reason'] is not None
            or not terminal['successful_owned_execution'] or not terminal['complete_result_present']
            or result['optimizer_steps'] != 0 or result['model_or_checkpoint_selected']
            or result['full_joint_Pred_or_Goal_accepted'] or result['resources']['CUDA_initialized']
            or result['final_Val_Test_original_pixels_new_nonKnown_GT_or_vocabulary_used']):
        raise ValueError('Actual normal full development, no optimizer/selection/scope expansion')
    if (not result['cohort']['all15_actual_complete_fixed_fits']
            or len(result['cohort']['trials']) != 15 or set(result['trials']) != set(FAMILIES)
            or set(result['Raw_reference']) != set(CAPS)):
        raise ValueError('All15 actual fixed fits and all six prefixes required')
    scope = result['development_scope']
    if (scope['physical_tracks'], scope['observations'], scope['metadata_frames']) != (472, 8753, 2562):
        raise ValueError('Full frozen physical population, no smaller denominator')
    if scope['physical_label_strata'] != {
            'Known': 302, 'declared_pseudoNovel': 119,
            'outside_eligible_Known': 2, 'no_retained_semantic_truth': 49}:
        raise ValueError('Unknown remains unlabeled, not fabricated Novel or background')
    table = {}
    for family in FAMILIES:
        seeds = result['trials'][family]
        if set(seeds) != set(SEEDS):
            raise ValueError('No missing seed or dev-selected endpoint')
        table[family] = {}
        for seed in SEEDS:
            if set(seeds[seed]['cases']) != set(CAPS):
                raise ValueError('Complete observed-prefix matrix required')
            prototype = seeds[seed]['prototype_scope']
            if (prototype['scope']['canonical_known_groups'] != 1158
                    or prototype['dev_labels_or_prototypes_or_old_checkpoints_used']):
                raise ValueError('Complete canonical offline fit5 prototypes only')
            probe = result['numeric_probes'][family][seed]
            if (probe['future_rows_mutated'] <= 0 or not probe['actual_first_prefix_future_invariant']
                    or not probe['actual_numeric_prefix_routing_rename_invariant']
                    or probe['temporal_benefit_or_Goal_accepted']):
                raise ValueError('Actual numeric prefix probes, not full online proof')
        for cap in CAPS:
            values = {}
            for seed in SEEDS:
                case = seeds[seed]['cases'][cap]
                if (case['Raw'] != result['Raw_reference'][cap]
                        or case['support']['full_physical_tracks'] != 472
                        or not case['support']['full_physical_gallery_short_and_unlabeled_not_dropped']
                        or case['feature_spaces']['zero_padding_or_undefined_cross_space_pointwise_cosine_used']):
                    raise ValueError('Identical full Raw footprint/native comparison, not padding')
                native = case['fresh_A']
                retrieval = native['retrieval']
                if (retrieval['physical_gallery_tracks'] != 472 or retrieval['declared_query_tracks'] != 119
                        or retrieval['declared_query_classes'] != 8 or not retrieval['samevideo_excluded']
                        or retrieval['unknown_gallery_declared_background_or_labeled_Novel']
                        or not retrieval['exact_tie_uniform_expectation_no_ID_ordering']
                        or native['Known']['Known_queries'] != 302):
                    raise ValueError('Frozen certified gallery/query populations')
                values[seed] = measurements(native)
                values[seed]['Raw_pairwise_Gram_MSE'] = case['Raw_drift']['full_pairwise_Gram_MSE']
            raw = measurements(result['Raw_reference'][cap])
            table[family][cap] = {
                key: statistics_by_seed({s: values[s][key] for s in SEEDS}) for key in values[SEEDS[0]]}
            table[family][cap]['paired_delta_against_zero_fit_Raw'] = {
                key: statistics_by_seed({s: values[s][key] - raw[key] for s in SEEDS}) for key in raw}
    dynamic_minus_static = {}
    for cap in CAPS:
        dynamic_minus_static[cap] = {
            key: statistics_by_seed({s: table['A2_EVIDENCE'][cap][key]['per_seed'][s]
                - table['A2_CAPACITY_CONTROL'][cap][key]['per_seed'][s] for s in SEEDS})
            for key in measurements(result['Raw_reference'][cap])}
    return {
        'status': 'POSTHOC_PUBLIC_ALL15_DEVELOPMENT_SUMMARY_NO_SELECTOR_OR_GOAL_ACCEPTANCE',
        'physical_tracks': 472, 'declared_pseudo_queries': 119, 'Known_queries': 302,
        'actual_fit_count': 15, 'evaluated_model_prefix_cases': 90,
        'Raw_reference_is_one_zero_fit_control_not_three_training_seeds': True,
        'Raw_reference': {cap: measurements(result['Raw_reference'][cap]) for cap in CAPS},
        'families': table, 'paired_A2_dynamic_minus_static': dynamic_minus_static,
        'sample_std_is_three_training_seeds_not_a_significance_or_video_CI_claim': True,
        'certified_retrieval_lower_bound_is_not_complete_semantic_purity_Hscore': True,
        'GT_track_diagnostic_not_Pred_physical_online_controller_or_locked_final': True,
        'optimizer_steps': 0, 'model_or_checkpoint_selected': False,
        'full_joint_Pred_or_Goal_accepted': False,
    }
