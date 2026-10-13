"""Public normal-completion evidence, never a grant to private data/weights."""
import json
import math
from src.trackocd_v2.io import sha256_file
from .raw_membership_formal_cohort import TRIALS, verified_cohort
from .raw_membership_development_protocol import PREFIX


def directory(control, seed):
    if (control, seed) not in TRIALS: raise ValueError('Only four predeclared endpoints')
    return f'{PREFIX}_{control}' + (f'_s{seed}' if seed is not None else '')


def trial_sources(control, seed):
    out = directory(control, seed)
    return (out+'/launch.json', out+'/result.json', out+'_supervisor_attempt_1.json')


def diagnostic_gate(d):
    n = d['lawful_groups']
    if (n <= 0 or d['Known_groups'] <= 0 or d['declared_pseudo_groups'] <= 0
            or d['Known_groups']+d['declared_pseudo_groups'] != n
            or not d['independent_Train_dev_no_optimizer_or_selection']
            or not d['declared_pseudo_is_existing_Known_derived_not_new_Novel_GT']
            or d['calibration_quality_proved_or_model_selected_from_diagnostics']
            or not d['declared_role_conditioning_and_class_weighting_not_full_deployment_prior']):
        raise ValueError('Complete conditional-lawful diagnostics, not selected/deployment posterior')
    for k in ('micro_NLL', 'class_macro_NLL', 'micro_Brier', 'class_macro_Brier',
              'fixed_10_bin_micro_ECE', 'fixed_10_bin_class_macro_ECE'):
        if not math.isfinite(d[k]) or d[k] < 0 or ('NLL' not in k and d[k] > 1):
            raise ValueError('Finite bounded original measured diagnostics')
    bins = d['bins']
    if (len(bins) != 10 or [b['bin_index'] for b in bins] != list(range(10))
            or sum(b['lawful_group_count'] for b in bins) != n
            or any(b['lawful_group_count'] < 0 or not 0 <= b['class_macro_mass'] <= 1+1e-12 for b in bins)
            or not math.isclose(sum(b['class_macro_mass'] for b in bins), 1., rel_tol=0, abs_tol=1e-12)):
        raise ValueError('All original fixed probability-bin counts/masses retained')


def verified_trial(root, control, seed):
    from scripts.trackocd_cvpr.evaluate_raw_membership_development import SOURCES
    paths = tuple(root/p for p in trial_sources(control, seed))
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError('Actual complete public launch/result/normal terminal required')
    launch, result, terminal = (json.loads(p.read_text()) for p in paths)
    if (result['status'] != 'MEASURED_FULL_RAW_MEMBERSHIP_DEV_NOT_SELECTED_JOINT_PRED_FINAL_OR_GOAL'
            or (result['control'], result['calendar_seed']) != (control, seed)
            or (launch['control'], launch['calendar_seed']) != (control, seed)
            or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None or not terminal['complete_result_present']
            or terminal['CPU_compute_workers'] != 1 or terminal['training_GPU_hours'] != 0
            or terminal['peak_sampled_worker_threads'] > 2 or terminal['peak_identified_native_driver_helpers'] > 1
            or terminal['peak_sampled_owned_worker_RSS_bytes'] > 3221225472
            or terminal['wall_seconds_including_worker_imports_network'] > 3000
            or result['resources']['CUDA_initialized'] or result['resources']['training_GPU_hours'] != 0
            or result['optimizer_steps'] != 0 or not result['all4_fits_and_all13_private_snapshot_fingerprints_checked']
            or result['dev_model_seed_threshold_bin_or_checkpoint_selection']
            or result['final_Val_Test_original_pixels_or_new_nonKnown_semantics_read']
            or result['foreground_Existing_gate_full_deployment_calibration_or_fair_latency_coverage_claimed']
            or result['full_semantic_truth_Hscore_CT_joint_Pred_final_or_Goal_accepted']
            or launch['live_remote_registration_proof'] != result['live_remote_registration_proof']
            or launch['preregistration_commit'] != result['live_remote_registration_proof']['sha']
            or set(launch['source_sha256']) != set(SOURCES)):
        raise ValueError('Actual bounded normal full evaluation, not partial/selected or expanded role')
    fit = next(r for r in verified_cohort(root)['trials'] if (r['control'], r['calendar_seed']) == (control, seed))
    if (launch['fixed_endpoint'] != fit['fixed_endpoint'] or result['evaluated_endpoint'] != fit['fixed_endpoint']
            or result['evaluated_model_parameter_sha256'] != fit['model_parameter_sha256']):
        raise ValueError('Same actual fixed formal parameters, not pilot/selected/warmstarted endpoint')
    scope = result['development_scope']
    if (scope['physical_tracks'], scope['observations'], scope['metadata_frames']) != (472, 8753, 2562):
        raise ValueError('Full uncapped physical scope')
    if (scope['physical_label_strata'] != {'Known':302, 'declared_pseudoNovel':119,
            'no_retained_semantic_truth':49, 'outside_eligible_Known':2}
            or scope['fit_policy_final_or_other_video_target_values_decoded']
            or scope['new_nonKnown_semantic_values_or_vocabulary_decoded']
            or not scope['external_evaluator_only_not_model_or_loss_input']):
        raise ValueError('Existing eligible targets only; all51 outside/unknown retained')
    reliability = result['conditional_reliability']; clock = reliability['full_clock']
    if ((clock['physical_observations'], clock['metadata_frames'], clock['empty_frames']) != (8753, 2562, 116)
            or sum(clock[k] for k in ('Known_groups', 'declared_pseudoNovel_groups', 'masked_groups')) != clock['canonical_groups']
            or not 0 < clock['canonical_groups'] <= 8753
            or not reliability['all_physical_tracks_and_unknowns_retained']
            or any(reliability[k] for k in ('unknown_conflicting_groups_as_negative_BG_Novel',
                'class_macro_or_role_conditional_task_is_deployment_prior', 'Existing_gate_foreground_purity_or_full_posterior_calibrated',
                'bin_threshold_seed_model_checkpoint_or_control_selection', 'private_score_target_tables_exported',
                'repeated_observations_prefixes_are_independent_replicates'))):
        raise ValueError('Complete diagnostic clock/masks, no fabricated deployment/independence')
    group = reliability['complete_group_observation_task']
    prefix = reliability['fixed_observed_prefix_track_tasks']
    if set(prefix) != {'1', '2', '4', '8', '16', 'all_available'}:
        raise ValueError('All fixed track-prefix tasks required')
    for pair in (group, *prefix.values()):
        for key in ('initial_fixed_Raw_task_score', 'fixed_formal_endpoint_task_score'):
            d = pair[key]; diagnostic_gate(d)
            if pair is group:
                if d['lawful_groups'] != clock['Known_groups']+clock['declared_pseudoNovel_groups']:
                    raise ValueError('Every eligible canonical group in diagnostic denominator')
            elif (pair['physical_tracks'] != 472 or pair['masked_tracks'] < 51
                    or d['lawful_groups']+pair['masked_tracks'] != 472
                    or d['Known_groups'] > 302 or d['declared_pseudo_groups'] > 119):
                raise ValueError('All physical track-prefix denominators, masked aliases/unknown retained')
    if (not result['actual_full_ID_rename_reverse_stream_equal']
            or not result['actual_future_probe']['verified_same_prior_events']
            or result['actual_future_probe']['future_observations_mutated'] <= 0
            or set(result['orders']) != {'1027', '1028', '1029'}):
        raise ValueError('ALL3 full orders and nonvacuous complete ID/reverse/future probes')
    for order, measured in result['orders'].items():
        if (measured['order_seed'] != int(order) or measured['observations'] != 8753
                or (measured['metadata_frames'], measured['empty_frames']) != (2562, 116)
                or measured['optimizer_steps'] != 0 or not measured['model_state_unchanged']
                or not measured['GT_oracle_control_not_Pred'] or measured['unknown_declared_background_or_Novel']
                or measured['self_support_exclusion_or_full_semantic_GT_Hscore_CT_or_Goal_acceptance']
                or not measured['prefix_slices_one_uncapped_causal_stream_not_separately_capped_replays']
                or sum(measured['physical_actions'].values()) != 8753
                or not set(measured['physical_actions']) <= {'KNOWN', 'NEW', 'EXISTING', 'WAIT'}
                or set(measured['prefix_snapshots']) != set(prefix)):
            raise ValueError('Every complete original causal order/physical action and prefix')
        for snapshot in measured['prefix_snapshots'].values():
            if (snapshot['physical_tracks'], snapshot['Known_queries'], snapshot['declared_pseudo_queries']) != (472, 302, 119):
                raise ValueError('Complete physical/Known/pseudo outcome denominator, no opportunity filtering')
    for p, h in launch['source_sha256'].items():
        if (root/p).is_symlink() or sha256_file(root/p) != h:
            raise ValueError('Unchanged actually registered numerical/public/source closure')
    return {'control':control, 'calendar_seed':seed, 'launch_sha256':sha256_file(paths[0]),
        'result_sha256':sha256_file(paths[1]), 'terminal_sha256':sha256_file(paths[2]),
        'all3_full_orders_and_complete_probes_verified':True, 'private_weights_or_GT_loaded_by_verifier':False,
        'full_deployment_calibration_fair_match_control_winner_or_Goal_accepted':False}


def verified_matrix(root):
    # A single complete trial never masquerades as the ALL4 scientific window.
    rows = [verified_trial(root, c, s) for c, s in TRIALS]
    originals = [json.loads((root/trial_sources(c, s)[1]).read_text()) for c, s in TRIALS]
    launches = [json.loads((root/trial_sources(c, s)[0]).read_text()) for c, s in TRIALS]
    baseline = originals[0]['conditional_reliability']
    for r, launch in zip(originals, launches):
        if launch['source_sha256'] != launches[0]['source_sha256']:
            raise ValueError('Same actual frozen numerical/source corpus across all4')
        q = r['conditional_reliability']
        if q['full_clock'] != baseline['full_clock']:
            raise ValueError('Same full physical diagnostic universe, not control-specific filtering')
        for key in ('1', '2', '4', '8', '16', 'all_available'):
            if q['fixed_observed_prefix_track_tasks'][key]['initial_fixed_Raw_task_score'] != baseline['fixed_observed_prefix_track_tasks'][key]['initial_fixed_Raw_task_score']:
                raise ValueError('Same input-only initial Raw reliability corpus before comparison')
        if q['complete_group_observation_task']['initial_fixed_Raw_task_score'] != baseline['complete_group_observation_task']['initial_fixed_Raw_task_score']:
            raise ValueError('Same complete group initial Raw diagnostic baseline')
        for order in ('1027', '1028', '1029'):
            if r['orders'][order]['private_video_ordinal_order_sha256'] != originals[0]['orders'][order]['private_video_ordinal_order_sha256']:
                raise ValueError('Same actual video ordinal schedules in all4')
    return {'all4_actual_complete_dev_trials_verified':True, 'trials':rows,
        'same_actual_initial_Raw_diagnostic_universe_and_video_orders':True,
        'fullpolicy_is_three_independent_training_seeds':False,
        'deployment_calibration_latency_coverage_winner_or_Goal_accepted':False}
