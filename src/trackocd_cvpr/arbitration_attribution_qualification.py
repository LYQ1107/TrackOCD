"""PUBLIC full-pilot completion/source/count qualification, not authority."""
import json
from src.trackocd_v2.io import sha256_file
from .arbitration_attribution_protocol import OUT, EXPECTED
from .raw_membership_formal_cohort import verified_cohort


def verify_actual_policy_pilot(root):
    from scripts.trackocd_cvpr.pilot_arbitration_attribution import SOURCES
    paths = (root/(OUT+'/launch.json'),root/(OUT+'/result.json'),root/(OUT+'_supervisor_attempt_1.json'))
    if any(not p.is_file() or p.is_symlink() for p in paths):raise ValueError('Complete original all5 policy result/launch/normal terminal first')
    launch,r,t = (json.loads(p.read_text()) for p in paths)
    if (r['status'] != 'MEASURED_READONLY_POLICY_ATTRIBUTION_PILOT_NOT_UNSEEN_DEV_INTERVENTION_WIN_OR_GOAL'
            or not t['successful_owned_execution'] or t['owned_worker_exit_code'] != 0
            or t['guard_stop_reason'] is not None or not t['complete_result_present']
            or t['CPU_compute_workers'] != 1 or t['peak_sampled_worker_threads'] > 2
            or t['peak_identified_native_driver_helpers'] > 1 or t['training_GPU_hours'] != 0
            or t['peak_sampled_owned_worker_RSS_bytes'] > EXPECTED['rss_bytes_ceiling']
            or t['wall_seconds_including_worker_imports_network'] > EXPECTED['wall_seconds_ceiling']
            or r['resources']['CUDA_initialized'] or r['resources']['training_GPU_hours'] != 0
            or not r['all13_private_snapshot_fingerprints_checked_before_policy'] or r['optimizer_steps'] != 0
            or r['shadow_or_GT_used_for_policy_memory_write'] or r['dev_final_Val_Test_pixels_or_new_nonKnown_semantics_read']
            or r['policy_resubstitution_is_unseen_or_joint_Goal_acceptance']
            or launch['preregistration_commit'] != r['live_remote_registration_proof']['sha']
            or launch['live_remote_registration_proof'] != r['live_remote_registration_proof']
            or set(launch['source_sha256']) != set(SOURCES) or set(r['models']) != set(EXPECTED['models'])):
        raise ValueError('Actual bounded normal all5 frozen model pilot, no expanded/selected or intervened history')
    formal = verified_cohort(root)
    expected_parameters = {'initial_Raw':'3ca6d4d867ba1958e6c100b771d1d5bbd432feda40c5085d761818e0ac86582e'}
    expected_parameters.update({('calendar'+str(x['calendar_seed']) if x['control'] == 'calendar_matched' else 'ONE_fullpolicy'):x['model_parameter_sha256'] for x in formal['trials']})
    schedule = None
    for name,result in r['models'].items():
        m = result['measured']; clock = m['full_clock']; strata = m['external_strata']
        if (result['formal_parameter_sha256'] != expected_parameters[name]
                or not result['normal_event_commit_state_parity'] or not result['actual_full_ID_reverse_outcome_event_state_equal']
                or not result['actual_future_prefix_equal'] or result['future_observations_mutated'] <= 0
                or not m['exact_numeric_support_witness_checked'] or not m['model_state_unchanged']
                or not m['same_history_one_decision_shadow_not_counterfactual_rollout_gain']
                or m['GT_or_shadow_decisions_used_for_any_policy_or_memory_write'] or m['order_seed'] != 1027
                or set(strata) != {'Known','policy_pseudo','masked'}
                or (clock['physical_observations'],clock['canonical_groups'],clock['metadata_frames'],clock['empty_frames']) != (1268,1268,374,1)
                or sum(clock.get(a,0) for a in ('KNOWN','NEW','EXISTING','WAIT')) != 1268):
            raise ValueError('Each complete actual policy clock/reference/ID/future/parameter witness, not a successful subset')
        if schedule is None:schedule = m['video_ordinal_schedule_sha256']
        if schedule != m['video_ordinal_schedule_sha256']:raise ValueError('Same actual complete policy video schedule per model')
        for kind,n in (('Known',791),('policy_pseudo',206),('masked',271)):
            s = strata[kind]
            if (s['canonical_groups'] != n or s['physical_observations'] != n
                    or sum(s.get(a,0) for a in ('KNOWN','NEW','EXISTING','WAIT')) != n
                    or any(type(v) is not int or v < 0 for v in s.values())):
                raise ValueError('All policy physical/eligible/masked group counts retained, no unknown negatives')
            existing = s.get('EXISTING',0)
            only = s.get('Existing_provenance_partition_only_own_groups',0)
            mixed = s.get('Existing_provenance_partition_own_and_other_groups',0)
            other = s.get('Existing_provenance_partition_other_only_groups',0)
            if (only+mixed+other != existing or s.get('Existing_only_own_groups',0) != only
                    or s.get('Existing_contains_own_groups',0) != only+mixed
                    or s.get('positive_named_gate_bypassed_groups',0) > min(existing,s.get('named_gate_positive_groups',0))
                    or s.get('same_history_remove_own_changes_action_groups',0) > existing
                    or s.get('same_history_remove_own_KNOWN_groups',0) > s.get('positive_named_gate_bypassed_groups',0)
                    or s.get('same_history_remove_own_correct_Known_groups',0) > s.get('same_history_remove_own_KNOWN_groups',0)
                    or kind != 'Known' and s.get('same_history_remove_own_correct_Known_groups',0) != 0):
                raise ValueError('Existing provenance partition and conditional one-step shadow count conservation')
        if m['actual_commits'] != clock.get('NEW',0)+clock.get('EXISTING',0):
            raise ValueError('Every actual anonymous commit preserved, no hypothetical score writes')
    for p,h in launch['source_sha256'].items():
        if (root/p).is_symlink() or sha256_file(root/p) != h:raise ValueError('Actual registered observer/cache/numeric/source path changed')
    return {'actual_normal_all5_policy_pilot_verified':True,'launch_sha256':sha256_file(paths[0]),
        'result_sha256':sha256_file(paths[1]),'terminal_sha256':sha256_file(paths[2]),
        'private_GT_models_or_row_tables_read_by_qualifier':False,
        'one_step_policy_resubstitution_is_unseen_dev_or_intervention_stream_win':False,
        'new_private_access_automatic_expansion_or_Goal_acceptance':False}
