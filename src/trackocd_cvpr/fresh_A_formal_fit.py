"""Public prerequisite and explicit single-trial configuration; no asset reads."""
import json
import os
from pathlib import Path
from src.trackocd_v2.io import sha256_file

CONFIG='configs/trackocd_cvpr/fresh_A_fixed200_execution_v1.json'
PILOT='outputs/trackocd_cvpr/cvpr2/fresh_A_fit5_optimizer_pilot_v1_CPU_visibility_retry1'
PREFIX='outputs/trackocd_cvpr/cvpr2/fresh_A_fit5_Fixed200_v1'
PILOT_SOURCES=(PILOT+'/result.json',PILOT+'/launch.json',PILOT+'_supervisor_attempt_1.json')


def configuration(root,family,seed):
    from .fresh_A_baseline import FAMILIES
    cfg=json.loads((Path(root)/CONFIG).read_text())
    expected={'status':'REGISTERED_FRESH_FIT5_FIXED200_SOURCE_ACTUAL_TRIALS_PENDING',
        'protocol':'configs/trackocd_cvpr/fresh_A_fit5_baselines_v1.json',
        'data_config':'configs/trackocd_cvpr/fit_R_optimizer_smoke_v1.json',
        'approval':'configs/trackocd_cvpr/formal_research_scope_approval_v1.json',
        'pilot_namespace':PILOT,'families':list(FAMILIES),'training_seeds':[1027,1028,1029],
        'optimizer_steps':200,'checkpoint_steps':[20,50,100,200],
        'checkpoint_selection':'Fixed200, never dev/final/Val/Test selection',
        'pilot_or_historical_checkpoint_warmstart':False,'execution_device':'CPU',
        'CPU_compute_workers':1,'rss_bytes_ceiling':3221225472,'wall_seconds_ceiling':3000,
        'output_bytes_ceiling':33554432,'output_prefix':PREFIX,
        'launch_one_explicit_family_seed_only':True,
        'parallel_fit_with_live_owned_M_eval_or_other_fresh_A_fit':False,
        'full_development_automatic_start':False,'formal15fits_dev_or_Goal_accepted':False,
        'pilot_result_sha256':'363e5a523db3e2ecd855a0afb77642cdd6c8862a5dd35505a6b4a60ca65dd0ce',
        'pilot_terminal_sha256':'fcaed77e08b583909e270c4cbf81176d6a3727806611cca7eedb870199bc9248',
        'pilot_launch_sha256':'fd96409e0543ec42201e1b56e9f6b123b07c70fcdcb0edf2549fd8438b7ae3b2'}
    if any(cfg.get(k)!=v for k,v in expected.items()) or family not in FAMILIES or type(seed) is not int or seed not in (1027,1028,1029):
        raise ValueError('Exact fixed200 family/seed/qualified pilot/safety protocol')
    return {**cfg,'family':family,'seed':seed,'output_directory':f'{PREFIX}_{family}_s{seed}'}


def verify_pilot(root,cfg):
    from .fresh_A_training import verify_matched_pilot
    from scripts.trackocd_cvpr.pilot_fresh_A_optimizer import SOURCES
    root=Path(root);p=root/PILOT
    for name,key in ((PILOT+'/result.json','pilot_result_sha256'),
                     (PILOT+'/launch.json','pilot_launch_sha256'),
                     (PILOT+'_supervisor_attempt_1.json','pilot_terminal_sha256')):
        f=root/name
        if f.is_symlink() or sha256_file(f)!=cfg[key]: raise ValueError('Actual immutable complete pilot prerequisite')
    result=json.loads((p/'result.json').read_text());launch=json.loads((p/'launch.json').read_text())
    terminal=json.loads((root/(PILOT+'_supervisor_attempt_1.json')).read_text())
    if (result['status']!='COMPLETED_FRESH_FIVE_FAMILY_FIT5_PILOT_NOT_FORMAL_DEV_OR_GOAL_ACCEPTANCE'
            or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code']!=0
            or terminal['guard_stop_reason'] is not None or not terminal['full_result_present']
            or terminal['CPU_compute_workers']!=1 or terminal['peak_identified_native_driver_helpers']>1
            or terminal['peak_sampled_worker_threads']>2 or terminal['training_GPU_hours']!=0
            or result['resources']['CUDA_initialized'] or result['formal_warmstart']
            or result['dev_final_Val_Test_inputs_used']
            or result['actual_parity']!=verify_matched_pilot(result['trials'])
            or result['live_remote_registration_proof']['sha']!='28ad15966c35fe893ab99a008d5423913db5a347'
            or launch['preregistration_commit']!=result['live_remote_registration_proof']['sha']):
        raise ValueError('All five actual normal pilots/parity/no dev or warmstart')
    # Retain the qualified numerical path, data loader and runtime guard;
    # do not silently change underlying source after a successful pilot.
    for path in SOURCES:
        if sha256_file(root/path)!=launch['source_sha256'].get(path):
            raise ValueError('Actual qualified pilot dependencies changed: '+path)
    return {'pilot_result_sha256':cfg['pilot_result_sha256'],'pilot_weights_read':False,
            'all_five_actual_qualified':True,'formal_initialization':'fresh same-seed adapter'}


def active_research_workers(root,*,proc_root=Path('/proc'),own_pid=None):
    root=Path(root); own_pid=os.getpid() if own_pid is None else own_pid;found=[]
    script=os.fsencode(str(root/'scripts/trackocd_cvpr/fit_fresh_A_fixed200.py'))
    m_prefix=os.fsencode(str(root/'scripts/trackocd_cvpr/evaluate_M_development'))
    for p in Path(proc_root).glob('[0-9]*/cmdline'):
        if int(p.parent.name)==own_pid: continue
        try: words=p.read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        is_M=any(w.startswith(m_prefix) and w.endswith(b'.py') for w in words)
        if script in words or (is_M and (b'--worker' in words or b'--supervisor' in words)):
            # Cmdline alone is not liveness: inspect kernel state too.
            try: state=(p.parent/'stat').read_text().rsplit(')',1)[1].split()[0]
            except (FileNotFoundError,PermissionError,ProcessLookupError): continue
            if state not in ('Z','X','x'):found.append(int(p.parent.name))
    return tuple(sorted(found))


def cohort_sources():
    from .fresh_A_baseline import FAMILIES
    return tuple(p for family in FAMILIES for seed in (1027,1028,1029)
        for p in (f'{PREFIX}_{family}_s{seed}/launch.json',f'{PREFIX}_{family}_s{seed}/result.json',
                  f'{PREFIX}_{family}_s{seed}_supervisor_attempt_1.json'))


def verified_cohort(root,*,check_private_endpoints=False):
    """All fifteen normal fixed endpoints before any full-dev baseline input.

    Public checks are necessary but not a new live registration proof. Caller
    must register ALL45 public artifacts plus source, gate fresh exact remote
    and only then enable private endpoint byte checks before dev data loading.
    """
    from .fresh_A_baseline import FAMILIES
    from scripts.trackocd_cvpr.fit_fresh_A_fixed200 import SOURCES
    root=Path(root);rows=[];adapters={};calendars={};qualified_sources=None
    for family in FAMILIES:
        for seed in (1027,1028,1029):
            out=root/f'{PREFIX}_{family}_s{seed}'
            paths=(out/'launch.json',out/'result.json',out.with_name(out.name+'_supervisor_attempt_1.json'))
            if any(not p.is_file() or p.is_symlink() for p in paths):raise ValueError('All15 actual immutable complete public trials required')
            launch,result,terminal=(json.loads(p.read_text()) for p in paths);trial=result['trial']
            if (result['status']!='COMPLETED_ONE_FRESH_FIXED200_FIT_NOT_ALL15_DEV_OR_GOAL_ACCEPTANCE'
                    or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code']!=0
                    or terminal['guard_stop_reason'] is not None or not terminal['full_result_present']
                    or terminal['last_persisted_completed_optimizer_steps']!=200
                    or terminal['training_GPU_hours']!=0 or terminal['CPU_compute_workers']!=1
                    or result['resources']['training_GPU_hours']!=0 or trial['training_GPU_hours']!=0
                    or terminal['peak_identified_native_driver_helpers']>1 or terminal['peak_sampled_worker_threads']>2
                    or trial['family']!=family or trial['seed']!=seed or trial['stage']!='Fixed200'
                    or trial['completed_optimizer_steps']!=200
                    or trial['parameter_count']!=(541889 if family.startswith('A2_') else 525056)
                    or trial['model_parameter_sha256_before']==trial['model_parameter_sha256_after']
                    or [r['step'] for r in trial['loss_curve']]!=list(range(1,201))
                    or result['pilot_or_historical_checkpoint_warmstart'] or result['dev_final_Val_Test_inputs_used']
                    or result['formal15fits_dev_or_Goal_accepted'] or result['resources']['CUDA_initialized']
                    or result['checkpoint_selection']!='Fixed200, never dev/final/Val/Test selection'
                    or trial['scope']['physical_tracks']!=1401 or trial['scope']['eligible_known_physical_tracks']!=1158
                    or trial['scope']['masked_no_fit5_target_tracks']!=243
                    or trial['scope']['supervised_classes']!=5
                    or launch['preregistration_commit']!=result['live_remote_registration_proof']['sha']
                    or launch['live_remote_registration_proof']!=result['live_remote_registration_proof']
                    or [r['step'] for r in result['private_checkpoints']]!=[20,50,100,200]
                    or not all(r['reopen_exact'] for r in result['private_checkpoints'])):
                raise ValueError('All fixed fits/normal actual terminals/full scope, no partial seeds or selection')
            if set(launch['source_sha256'])!=set(SOURCES):raise ValueError('Every qualified training dependency required')
            if qualified_sources is None:qualified_sources=launch['source_sha256']
            if launch['source_sha256']!=qualified_sources:raise ValueError('All15 exact same qualified training source')
            for lookup,key in ((adapters,'adapter_initialization_sha256'),(calendars,'batch_calendar_sha256')):
                if seed in lookup and lookup[seed]!=trial[key]:raise ValueError('Actual shared initialization/calendar per seed, not config assertion')
                lookup[seed]=trial[key]
            endpoint=next(r for r in result['private_checkpoints'] if r['step']==200)
            if check_private_endpoints:
                f=out/'private_step200_checkpoint.pt'
                if f.is_symlink() or f.stat().st_size!=endpoint['bytes'] or sha256_file(f)!=endpoint['sha256']:
                    raise ValueError('Actual frozen private Fixed200 endpoint bytes differ')
            rows.append({'family':family,'seed':seed,'result_sha256':sha256_file(paths[1]),
                         'terminal_sha256':sha256_file(paths[2]),'endpoint':endpoint})
    if len(set(adapters.values()))!=3 or len(set(calendars.values()))!=3:
        raise ValueError('Three actually distinct seeded initializations/calendars')
    for path,expected in qualified_sources.items():
        if (root/path).is_symlink() or sha256_file(root/path)!=expected:raise ValueError('Frozen actual cohort training dependency changed')
    return {'all15_actual_complete_fixed_fits':True,'same_seed_initialization_and_calendar_verified':True,
            'three_distinct_actual_seeds_verified':True,'trials':rows,'full_dev_or_Goal_accepted':False,
            'private_endpoint_hashes_checked':check_private_endpoints}
