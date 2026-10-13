"""Public exact actual dual-pilot prerequisite, not permission or quality."""
import json
from src.trackocd_v2.io import sha256_file

NS = 'outputs/trackocd_cvpr/cvpr3/raw_NamedKnown_membership_dual_pilot50_v1'
HASHES = {NS+'/launch.json':'51e36d502d89d753f0cce56e39b0770c72f919c4e7ca65c0097f24a7166a09c5',
    NS+'/result.json':'6ca65d67e4fc68a565009f8054672bdeae35db672ca2f23d39973f7a1451df7b',
    NS+'_supervisor_attempt_1.json':'7463b69ff7109b6792de64b11201a31bca5b599ce381d33e01f5a5873352e6ab'}


def verify_actual_pilot(root):
    from scripts.trackocd_cvpr.pilot_raw_known_membership_controls import SOURCES
    for p,h in HASHES.items():
        if not (root/p).is_file() or (root/p).is_symlink() or sha256_file(root/p)!=h:
            raise ValueError('Exact original actual dual-pilot public artifact fingerprints')
    launch = json.loads((root/(NS+'/launch.json')).read_text())
    r = json.loads((root/(NS+'/result.json')).read_text())
    t = json.loads((root/(NS+'_supervisor_attempt_1.json')).read_text())
    c = r['calendar_control']; f = r['full_policy_control']; collection = r['full_policy_collection']
    if (not t['successful_owned_execution'] or t['owned_worker_exit_code']!=0 or t['guard_stop_reason'] is not None
            or not t['complete_result_present'] or t['CPU_compute_workers']!=1 or t['training_GPU_hours']!=0
            or t['peak_sampled_worker_threads']>2 or t['peak_identified_native_driver_helpers']>1
            or t['peak_sampled_owned_worker_RSS_bytes']>3221225472 or t['wall_seconds_including_worker_imports_network']>3000
            or r['status']!='COMPLETED_FIXED_RAW_MEMBERSHIP_DUAL_PILOT_NOT_DEV_CALIBRATION_FOREGROUND_OR_GOAL'
            or not r['operational_prerequisites_for_potential_later_registration']
            or r['actual_calibration_generalization_or_control_winner_selected']
            or r['pilot_warmstart_or_automatic_formal_dev_start'] or r['dev_final_Val_Test_or_new_nonKnown_inputs_used']
            or r['raw_private_GT_scalar_table_weights_or_traces_exported']
            or r['Existing_gate_foreground_purity_or_deployment_posterior_calibrated']
            or r['resources']['CUDA_initialized'] or r['resources']['training_GPU_hours']!=0
            or not r['initial_NamedKnown_gate_exact_on_actual_scores']
            or c['optimizer_steps']!=50 or c['trainable_parameters']!=2
            or c['initialization_sha256']==c['endpoint_parameter_sha256']
            or c['actual_shared_calendar_sha256']!='1fd8b98bc798aa090c92c7f15ea218f0730f78b5e8e2addef49a05bc88dadb6d'
            or not f['solver_success'] or f['solver_status']!=0 or f['iterations']!=12 or f['function_evaluations']!=13
            or f['initial_gradient_norm']<=0 or f['projected_gradient_infinity_norm']>1e-8
            or f['trainable_parameters']!=2 or f['training_seeds_or_calendar_matched_to_C']
            or f['parameter_sha256']==c['initialization_sha256']):
        raise ValueError('Actual complete finite-gradient full-clock converged dual pilot only, no quality or authority claim')
    expected = {'Known_groups':791,'policy_pseudo_groups':206,'masked_groups':271,'canonical_groups':1268,
                'physical_observations':1268,'metadata_frames':374,'empty_frames':1,'supervised_classes':5}
    if (any(collection[k]!=v for k,v in expected.items())
            or collection['unknown_as_negative_BG_Novel_or_individual_anonymous_target']
            or [x['step'] for x in c['loss_curve']]!=list(range(1,51))
            or any(x['gradient_norm_before_clip']<=0 for x in c['loss_curve'])
            or sum(x['lawful_group_counts'].get('Known',0) for x in c['loss_curve'])!=119
            or sum(x['lawful_group_counts'].get('policy_pseudo',0) for x in c['loss_curve'])!=26
            or c['episodes']!=[{'empty_frames':1,'masked_groups':271,'metadata_frames':374,'physical_observations':1268}]):
        raise ValueError('Real role exposure/masked physical inputs/all50steps, no unknown negatives')
    if (launch['preregistration_commit']!='0c9d3b7998dd19c5f8267e47116e2c43e66b70d3'
            or launch['live_remote_registration_proof']!=r['live_remote_registration_proof']
            or launch['live_remote_registration_proof']['sha']!=launch['preregistration_commit']
            or set(launch['source_sha256'])!=set(SOURCES)):
        raise ValueError('Actual original live source registration before private inputs')
    for p,h in launch['source_sha256'].items():
        if (root/p).is_symlink() or sha256_file(root/p)!=h:
            raise ValueError('Actual qualified numerical/input/checkpoint/runtime source changed')
    checkpoints = r['private_checkpoints']
    if (len(checkpoints)!=2 or [x['control'] for x in checkpoints]!=['calendar_matched','full_policy']
            or not all(x['reopen_exact'] for x in checkpoints) or checkpoints[0]['step']!=50
            or checkpoints[1]['step']!=12 or not checkpoints[0]['AdamW_and_RNG_reopened']
            or checkpoints[1]['AdamW_and_RNG_reopened']
            or any(x['solver_internal_LBFGS_resume_state_claimed'] for x in checkpoints)):
        raise ValueError('Exact own distinct checkpoints, no fabricated solver resume state')
    for d in r['resubstitution_diagnostics_on_same_training_samples'].values():
        if (d['lawful_groups']!=997 or d['Known_groups']!=791 or d['policy_pseudo_groups']!=206
                or not d['declared_role_conditioning_and_class_weighting_not_full_deployment_prior']
                or d['calibration_quality_proved_or_model_selected_from_diagnostics']):
            raise ValueError('Same-training lawful-role diagnostics, not transfer/calibration acceptance')
    return {'actual_normal_dual_pilot_prerequisites_checked':True,'result_sha256':HASHES[NS+'/result.json'],
        'private_data_or_weights_read':False,'formal_warmstart_auto_start_or_Goal_accepted':False,
        'calibration_quality_or_deployment_posterior_claimed':False}
