"""Actual informative pilot prerequisite only; no formal/development authority."""
import json
from src.trackocd_v2.io import sha256_file
from scripts.trackocd_cvpr.pilot_spread_known_rejection import SOURCES

NS='outputs/trackocd_cvpr/cvpr3/paired_spread_Known_rejection_pilot50_v1'
HASHES={NS+'/result.json':'aff6bc17536e538a4c0833e18d075e8d965abbb55e7288dbda900aea48283e79',
    NS+'/launch.json':'963ed084987263f3109eb29ae0ad34c80d0b7e574c2b2d01d839b91983adbd80',
    NS+'_supervisor_attempt_1.json':'7ba2f6a05e89ca98000c91ba9fd37af5edc2553bd5e895136f0d2d0dec00c6d4'}


def verify_informative_pilot(root):
    for path,digest in HASHES.items():
        if (root/path).is_symlink() or sha256_file(root/path)!=digest:
            raise ValueError('Exact one original informative spread pilot artifacts')
    r=json.loads((root/(NS+'/result.json')).read_text())
    l=json.loads((root/(NS+'/launch.json')).read_text())
    t=json.loads((root/(NS+'_supervisor_attempt_1.json')).read_text()); s=r['summary']
    if (not t['successful_owned_execution'] or t['owned_worker_exit_code']!=0 or t['guard_stop_reason'] is not None
            or not t['complete_result_present'] or t['CPU_compute_workers']!=1 or t['training_GPU_hours']!=0
            or t['peak_sampled_owned_worker_RSS_bytes']>3221225472
            or t['wall_seconds_including_worker_imports_network']>3000
            or t['peak_sampled_worker_threads']>2 or t['peak_identified_native_driver_helpers']>1
            or r['status']!='COMPLETED_SPREAD_PAIRED50_PILOT_NOT_FORMAL_DEV_CALIBRATION_OR_GOAL_ACCEPTANCE'
            or not r['actual_nonzero_rejection_gradient']
            or not r['pilot_informative_for_potential_later_formal_registration']
            or r['pilot_warmstart_formal_automatic_start_or_dev_selection'] or r['dev_final_Val_Test_inputs_used']
            or r['resources']['CUDA_initialized'] or s['optimizer_steps_each']!=50
            or s['trainable_parameters_each']!=708 or len(set(s['initialization_sha256'].values()))!=1
            or len(set(s['endpoint_parameter_sha256'].values()))!=2
            or s['offline_eligible_Known_frames_each']!=[333]
            or s['actual_shared_physical_calendar_sha256']!='1fd8b98bc798aa090c92c7f15ea218f0730f78b5e8e2addef49a05bc88dadb6d'):
        raise ValueError('Actual normal full paired50 with extra objective exercised, not uninformative completion')
    if l['preregistration_commit']!='5fc8847ce3bc2e0e01ee5733a1c31ebf49ff8b68' or l['live_remote_registration_proof']!=r['live_remote_registration_proof']:
        raise ValueError('Original exact live worker registration')
    for path in SOURCES:
        if sha256_file(root/path)!=l['source_sha256'].get(path): raise ValueError('Qualified numeric/source path changed')
    for arm in s['arms']:
        e=s['episodes'][arm][0]; rows=s['loss_curves'][arm]
        if (e['metadata_frames'],e['physical_observations'],e['empty_frames'],e['videos'])!=(374,1268,1,10):
            raise ValueError('Complete policy physical clock, no subset')
        if [r['step'] for r in rows]!=list(range(1,51)) or any(r['rejection_coefficient']!=(.25 if arm=='Known_set_rejection' else 0.) for r in rows):
            raise ValueError('Actual fixed objective/budget')
    if sum(r['rejection_parameter_gradient_norm']>0 for r in s['loss_curves']['Known_set_rejection'])!=26:
        raise ValueError('Real positive extra-gradient receipt, not tail-count substitution')
    if len(r['private_checkpoints'])!=2 or not all(c['step']==50 and c['reopen_exact'] for c in r['private_checkpoints']):
        raise ValueError('Actual own model/AdamW/RNG snapshots reopened')
    return {'actual_informative_pilot_checked':True,'result_sha256':HASHES[NS+'/result.json'],
        'private_weights_read_or_warmstart':False,'formal_dev_calibration_or_Goal_accepted':False}
