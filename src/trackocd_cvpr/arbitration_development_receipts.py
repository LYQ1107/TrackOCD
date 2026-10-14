"""Public normal full-dev observer qualification, never partial acceptance."""
import json
from src.trackocd_v2.io import sha256_file
from .arbitration_development_protocol import PREFIX,EXPECTED
from .raw_membership_formal_cohort import verified_cohort


def trial_sources(model):
    if model not in EXPECTED['models']:raise ValueError('All5 fixed model enum only')
    base = PREFIX+'_'+model
    return (base+'/launch.json',base+'/result.json',base+'_supervisor_attempt_1.json')


def verified_trial(root,model):
    from scripts.trackocd_cvpr.evaluate_arbitration_development import SOURCES
    paths = tuple(root/p for p in trial_sources(model))
    if any(not p.is_file() or p.is_symlink() for p in paths):raise ValueError('Complete original full-dev result/launch/normal terminal required')
    launch,r,t = (json.loads(p.read_text()) for p in paths)
    if (r['status'] != 'MEASURED_COMPLETE_TRAIN_DEV_ARBITRATION_OBSERVER_NOT_INTERVENED_STREAM_JOINT_PRED_FINAL_OR_GOAL'
            or r['model'] != model or launch['model'] != model
            or not t['successful_owned_execution'] or t['owned_worker_exit_code'] != 0 or t['guard_stop_reason'] is not None
            or not t['complete_result_present'] or t['CPU_compute_workers'] != 1 or t['training_GPU_hours'] != 0
            or t['peak_sampled_worker_threads'] > 2 or t['peak_identified_native_driver_helpers'] > 1
            or t['peak_sampled_owned_worker_RSS_bytes'] > EXPECTED['rss_bytes_ceiling']
            or t['wall_seconds_including_worker_imports_network'] > EXPECTED['wall_seconds_ceiling']
            or r['resources']['CUDA_initialized'] or r['resources']['training_GPU_hours'] != 0
            or not r['all13_snapshot_fingerprints_before_dev'] or not r['full_ID_reverse_events_outcomes_state_equal']
            or not r['actual_future_probe']['nonempty_prior_events_equal'] or r['actual_future_probe']['future_observations_mutated'] <= 0
            or not r['external_bucket_policy_pseudo_is_declared_dev_pseudo_not_training']
            or r['optimizer_steps'] != 0 or r['shadow_or_GT_used_for_policy_memory_write']
            or r['dev_head_seed_order_threshold_selected'] or r['final_Val_Test_original_pixels_or_new_nonKnown_semantics_read']
            or r['one_decision_shadows_are_full_intervention_gain_or_Goal_acceptance']
            or launch['live_remote_registration_proof'] != r['live_remote_registration_proof']
            or launch['preregistration_commit'] != r['live_remote_registration_proof']['sha']
            or set(launch['source_sha256']) != set(SOURCES) or set(r['orders']) != {'1027','1028','1029'}):
        raise ValueError('Actual bounded normal complete frozen-model full clocks/references/probes, not partial or intervened')
    formal = verified_cohort(root)
    expected = {'initial_Raw':'3ca6d4d867ba1958e6c100b771d1d5bbd432feda40c5085d761818e0ac86582e'}
    expected.update({('calendar'+str(x['calendar_seed']) if x['control'] == 'calendar_matched' else 'ONE_fullpolicy'):x['model_parameter_sha256'] for x in formal['trials']})
    if r['evaluated_parameter_sha256'] != expected[model]:raise ValueError('Actual fixed formal endpoint, not selected weight')
    scope = r['development_scope']
    if ((scope['physical_tracks'],scope['observations'],scope['metadata_frames']) != (472,8753,2562)
            or scope['physical_label_strata'] != {'Known':302,'declared_pseudoNovel':119,'outside_eligible_Known':2,'no_retained_semantic_truth':49}
            or scope['new_nonKnown_semantic_values_or_vocabulary_decoded'] or scope['fit_policy_final_or_other_video_target_values_decoded']
            or not scope['external_evaluator_only_not_model_or_loss_input']):
        raise ValueError('Complete lawful physical/semantic dev denominators, all51 unknown/outside retained')
    for order,result in r['orders'].items():
        m = result['measured']; c = m['full_clock']; s = m['external_strata']
        if (not result['actual_full_normal_event_commit_state_equal'] or m['order_seed'] != int(order)
                or not m['exact_numeric_support_witness_checked'] or not m['model_state_unchanged']
                or not m['same_history_one_decision_shadow_not_counterfactual_rollout_gain']
                or m['GT_or_shadow_decisions_used_for_any_policy_or_memory_write']
                or (c['canonical_groups'],c['physical_observations'],c['metadata_frames'],c['empty_frames']) != (8753,8753,2562,116)
                or set(s) != {'Known','policy_pseudo','masked'}
                or sum(c.get(k,0) for k in ('KNOWN','NEW','EXISTING','WAIT')) != 8753):
            raise ValueError('Each complete full-order original reference/observer/clock, not subset')
        for kind,n in (('Known',4808),('policy_pseudo',2743),('masked',1202)):
            a = s[kind]
            if (a['canonical_groups'] != n or a['physical_observations'] != n
                    or sum(a.get(k,0) for k in ('KNOWN','NEW','EXISTING','WAIT')) != n
                    or any(type(v) is not int or v < 0 for v in a.values())):
                raise ValueError('All eligible/masked physical dev observations retained')
            existing = a.get('EXISTING',0); own = a.get('Existing_provenance_partition_only_own_groups',0)
            mixed = a.get('Existing_provenance_partition_own_and_other_groups',0); other = a.get('Existing_provenance_partition_other_only_groups',0)
            if (own+mixed+other != existing or a.get('Existing_only_own_groups',0) != own
                    or a.get('Existing_contains_own_groups',0) != own+mixed
                    or a.get('positive_named_gate_bypassed_groups',0) > min(existing,a.get('named_gate_positive_groups',0))
                    or a.get('same_history_remove_own_changes_action_groups',0) > existing
                    or a.get('same_history_remove_own_KNOWN_groups',0) > a.get('positive_named_gate_bypassed_groups',0)
                    or a.get('same_history_remove_own_correct_Known_groups',0) > a.get('same_history_remove_own_KNOWN_groups',0)
                    or kind != 'Known' and a.get('same_history_remove_own_correct_Known_groups',0) != 0):
                raise ValueError('Support provenance and same-history conditional count conservation')
        if m['actual_commits'] != c.get('NEW',0)+c.get('EXISTING',0):raise ValueError('All actual prediction-owned commits retained')
    for p,h in launch['source_sha256'].items():
        if (root/p).is_symlink() or sha256_file(root/p) != h:raise ValueError('Actual registered numerical/adapter/source closure changed')
    return {'model':model,'launch_sha256':sha256_file(paths[0]),'result_sha256':sha256_file(paths[1]),
        'terminal_sha256':sha256_file(paths[2]),'all3_full_orders_each_original_reference_and_full_probes_checked':True,
        'private_models_GT_or_row_tables_read':False,'full_intervention_causal_gain_or_Goal_accepted':False}


def verified_matrix(root):
    rows = [verified_trial(root,m) for m in EXPECTED['models']]
    originals = [json.loads((root/trial_sources(m)[1]).read_text()) for m in EXPECTED['models']]
    launches = [json.loads((root/trial_sources(m)[0]).read_text()) for m in EXPECTED['models']]
    reference = originals[0]
    for result,launch in zip(originals,launches):
        if launch['source_sha256'] != launches[0]['source_sha256']:raise ValueError('Same actual source closure for all5 observed models')
        if result['development_scope'] != reference['development_scope'] or result['offline_fit_prior_scope'] != reference['offline_fit_prior_scope']:
            raise ValueError('Same actual independent dev and offline fit prior universe')
        for o in ('1027','1028','1029'):
            if result['orders'][o]['measured']['video_ordinal_schedule_sha256'] != reference['orders'][o]['measured']['video_ordinal_schedule_sha256']:
                raise ValueError('Same actual ordinal schedule per-order across all5')
    return {'all5_actual_complete_Train_dev_attribution_verified':True,'trials':rows,
        'one_step_shadows_are_full_intervention_stream_or_Goal_gain':False,
        'private_GT_model_data_loaded':False}
