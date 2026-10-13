"""Public four-fit proof; endpoint bytes opt-in after later source/authority gates."""
import json
from src.trackocd_v2.io import sha256_file
from .raw_membership_formal_protocol import PREFIX

TRIALS=(('calendar_matched',1027),('calendar_matched',1028),('calendar_matched',1029),('full_policy',None))


def directory(control,seed):
    if (control,seed) not in TRIALS: raise ValueError('Explicit four-control cohort')
    return f'{PREFIX}_{control}'+(f'_s{seed}' if seed is not None else '')


def cohort_sources():
    return tuple(p for c,s in TRIALS for p in (directory(c,s)+'/launch.json',directory(c,s)+'/result.json',directory(c,s)+'_supervisor_attempt_1.json'))


def verified_cohort(root,*,check_private_snapshots=False):
    from scripts.trackocd_cvpr.fit_raw_known_membership_formal import SOURCES
    rows=[];source=None;calendars={};initials=set();tables=set();private_snapshots=[]
    for control,seed in TRIALS:
        out=root/directory(control,seed)
        paths=(out/'launch.json',out/'result.json',out.with_name(out.name+'_supervisor_attempt_1.json'))
        if any(not p.is_file() or p.is_symlink() for p in paths):
            raise ValueError('ALL4 actual normal complete fresh formal control trials required')
        l,r,t=(json.loads(p.read_text()) for p in paths);s=r['summary'];saved=r['private_checkpoints'];collection=r['full_policy_collection']
        if (r['status']!='COMPLETED_FRESH_FORMAL_RAW_MEMBERSHIP_CONTROL_NOT_DEV_CALIBRATION_OR_GOAL'
                or r['control']!=control or r['calendar_seed']!=seed
                or r['old_or_pilot_weight_warmstart'] or r['dev_final_Val_Test_or_new_nonKnown_inputs_used']
                or r['solver_resume_state_transfer_calibration_or_control_winner_claimed'] or r['all4formal_fits_dev_or_Goal_accepted']
                or r['resources']['CUDA_initialized'] or r['resources']['training_GPU_hours']!=0
                or not t['successful_owned_execution'] or t['owned_worker_exit_code']!=0 or t['guard_stop_reason'] is not None
                or not t['complete_result_present'] or t['CPU_compute_workers']!=1 or t['training_GPU_hours']!=0
                or t['peak_sampled_worker_threads']>2 or t['peak_identified_native_driver_helpers']>1
                or t['peak_sampled_owned_worker_RSS_bytes']>3221225472 or t['wall_seconds_including_worker_imports_network']>3000
                or s['trainable_parameters']!=2 or r['model_parameter_sha256']==r['initialization_sha256']
                or l['control']!=control or l['calendar_seed']!=seed
                or l['preregistration_commit']!=r['live_remote_registration_proof']['sha']
                or l['live_remote_registration_proof']!=r['live_remote_registration_proof']
                or set(l['source_sha256'])!=set(SOURCES)
                or any(not x['reopen_exact'] or x['solver_internal_resume_state_claimed'] for x in saved)):
            raise ValueError('Actual complete bounded fresh control/normal receipt, not selected or pilot weights')
        if source is None:source=l['source_sha256']
        if l['source_sha256']!=source:raise ValueError('Same actual frozen qualified numerical/source path in all4')
        expected={'Known_groups':791,'policy_pseudo_groups':206,'masked_groups':271,'canonical_groups':1268,
            'physical_observations':1268,'metadata_frames':374,'empty_frames':1,'supervised_classes':5}
        if any(collection[k]!=v for k,v in expected.items()) or collection['unknown_as_negative_BG_Novel_or_individual_anonymous_target']:
            raise ValueError('Complete shared lawful scalar universe; no unknown negatives or data drop')
        tables.add(collection['private_scalar_target_weight_table_sha256']);initials.add(r['initialization_sha256'])
        if control=='calendar_matched':
            if (s['optimizer_steps']!=200 or [x['step'] for x in s['loss_curve']]!=list(range(1,201))
                    or len(s['episodes'])!=4 or [x['step'] for x in saved]!=[20,50,100,200]
                    or any(not x['AdamW_and_RNG_reopened'] for x in saved) or r['solver_converged'] is not None
                    or s['endpoint_parameter_sha256']!=r['model_parameter_sha256']
                    or any(x['gradient_norm_before_clip']<=0 for x in s['loss_curve'])
                    or any((e['physical_observations'],e['metadata_frames'],e['empty_frames'],e['masked_groups'])!=(1268,374,1,271) for e in s['episodes'])):
                raise ValueError('All200updates/four full clocks/four true model+AdamW+RNG snapshots')
            old=root/f'outputs/trackocd_cvpr/cvpr3/paired_spread_Known_rejection_Fixed200_v1_s{seed}/result.json'
            actual=json.loads(old.read_text())['summary']['actual_shared_physical_calendar_sha256']
            if s['actual_shared_calendar_sha256']!=actual:raise ValueError('Actual shared C frame update calendar perseed')
            calendars[seed]=actual
        else:
            if (not r['solver_converged'] or not s['solver_success'] or s['solver_status']!=0
                    or s['iterations']>200 or s['function_evaluations']>1000 or s['initial_gradient_norm']<=0
                    or s['training_seeds_or_calendar_matched_to_C'] or s['parameter_sha256']!=r['model_parameter_sha256']
                    or len(saved)!=1 or saved[0]['step']!=s['iterations'] or saved[0]['AdamW_and_RNG_reopened']):
                raise ValueError('ONE actual bounded converged fullpolicy fit, not artificial3seeds or solver resume')
        for d in r['resubstitution_diagnostics_on_same_training_samples'].values():
            if (d['lawful_groups']!=997 or d['calibration_quality_proved_or_model_selected_from_diagnostics']
                    or not d['declared_role_conditioning_and_class_weighting_not_full_deployment_prior']):
                raise ValueError('Scoped same-data diagnostic only, no calibrated transfer claim')
        for snapshot in saved:
            p=out/(f"private_step{snapshot['step']}_checkpoint.pt" if control=='calendar_matched' else 'private_full_policy_checkpoint.pt')
            private_snapshots.append((p,snapshot))
        rows.append({'control':control,'calendar_seed':seed,'result_sha256':sha256_file(paths[1]),
            'launch_sha256':sha256_file(paths[0]),'terminal_sha256':sha256_file(paths[2]),
            'model_parameter_sha256':r['model_parameter_sha256'],'fixed_endpoint':saved[-1],
            'actual_checkpoint_count':len(saved)})
    if len(initials)!=1 or len(tables)!=1 or len(set(calendars.values()))!=3:
        raise ValueError('Same deterministic initial and scalar corpus, three distinct actual C calendars')
    for p,h in source.items():
        if (root/p).is_symlink() or sha256_file(root/p)!=h:raise ValueError('Actual formal control numerical/source dependencies changed')
    # All four public trial proofs and source closure pass before ANY private
    # file stat/hash, even if a later trial has an invalid solver/normal receipt.
    if check_private_snapshots:
        for p,snapshot in private_snapshots:
            if not p.is_file() or p.is_symlink() or p.stat().st_size!=snapshot['bytes'] or sha256_file(p)!=snapshot['sha256']:
                raise ValueError('Actual own fixed private snapshot byte fingerprints')
    return {'all4_actual_fresh_formal_controls_verified':True,'trials':rows,
        'actual_three_distinct_matched_C_calendars_verified':True,'same_deterministic_initialization_and_scalar_table':True,
        'fullpolicy_is_three_training_seeds':False,'private_snapshot_bytes_checked':check_private_snapshots,
        'transfer_calibration_control_winner_or_Goal_accepted':False}
