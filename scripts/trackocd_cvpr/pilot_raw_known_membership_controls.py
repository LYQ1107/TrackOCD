#!/usr/bin/env python3
"""One explicit dual-control pilot; source/authority/cache gates before inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration
from src.trackocd_cvpr.raw_known_membership_protocol import source_window,CONFIG,MATRIX
from src.trackocd_cvpr.spread_C_development_complete_matrix import trial_sources,ARMS,SEEDS
from scripts.trackocd_cvpr import evaluate_spread_C_development as prior

SCRIPT = 'scripts/trackocd_cvpr/pilot_raw_known_membership_controls.py'
OUT = 'outputs/trackocd_cvpr/cvpr3/raw_NamedKnown_membership_dual_pilot50_v1'
SOURCES = tuple(dict.fromkeys((*prior.SOURCES,*(p for a in ARMS for s in SEEDS for p in trial_sources(a,s)),
    'src/trackocd_cvpr/spread_C_development_complete_matrix.py',CONFIG,MATRIX,SCRIPT,
    'src/trackocd_cvpr/raw_known_membership_protocol.py','src/trackocd_cvpr/raw_known_membership_control.py',
    'src/trackocd_cvpr/fresh_A_training.py','src/trackocd_cvpr/fresh_A_baseline.py',
    'src/trackocd_core/train_first_representation.py','src/trackocd_core/representation.py',
    'outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1/result.json',
    'outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1_supervisor_attempt_1.json')))


def configuration(root):
    control = source_window(root)
    return {**json.loads((root/control['data_config']).read_text()),**control,
            'approval':control['authority'],'output_directory':OUT}


def source_gate(args,cfg,safety):
    if cfg != configuration(ROOT): raise ValueError('Exact one fixed dual-control pilot')
    proof = verify_registration(ROOT,safety['branch'],args.preregistration_commit,args.git_proxy,allow_public_api=True)
    for p in SOURCES:
        if (ROOT/p).is_symlink() or subprocess.check_output(['git','show',args.preregistration_commit+':'+p],cwd=ROOT)!=(ROOT/p).read_bytes():
            raise ValueError('All actual full6 source/public prerequisites registered before private input')
    a = json.loads((ROOT/cfg['approval']).read_text())
    if (a['status']!='USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN' or a['video_scope']!=407
            or a['allowed_training_label_roles']!=['fit_known','policy_train_pseudo']
            or a['new_nonKnown_semantic_GT_or_Novel_vocabulary_authorized']
            or a['unlabeled_objects_background_or_inferred_Novel_labels']
            or a['old_51_heldout_Val_Test_DEVplus_Q1_public_new_sealed_and_other_videos_allowed']
            or a['foreign_process_interference_allowed']):
        raise ValueError('Real human-approved scoped fit/policy roles, no new labels or process authority')
    q = ROOT/'outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1'
    t = json.loads(q.with_name(q.name+'_supervisor_attempt_1.json').read_text())
    if (sha256_file(q/'result.json')!=cfg['policy_input_qualification_sha256']
            or not t['successful_owned_execution'] or t['owned_worker_exit_code']!=0 or t['guard_stop_reason'] is not None):
        raise ValueError('Actual exact full-policy input qualification before private load')
    return proof


def busy():
    if prior.busy(): return True
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name)==os.getpid(): continue
        try:
            words = p.read_bytes().split(b'\0'); state = (p.parent/'stat').read_text().rsplit(')',1)[1].split()[0]
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        if os.fsencode(str(ROOT/SCRIPT)) in words and state not in ('Z','X','x'): return True
    return False


def supervise(args,cfg,safety):
    out = ROOT/OUT; terminal = out.with_name(out.name+'_supervisor_attempt_1.json')
    if out.exists() or terminal.exists(): raise ValueError('One actual pilot attempt only; preserve all prior outputs')
    if busy(): raise RuntimeError('Observe live owned research work, no concurrent pilot')
    source_gate(args,cfg,safety); start = time.monotonic(); peak = threads = helpers = 0; reason = None
    mem = {k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    if (prior.runtime.resources_ok(cfg,safety,ROOT,start,0)
            or mem['MemAvailable']<safety['system_ram_reserve_fraction']*mem['MemTotal']+cfg['rss_bytes_ceiling']):
        raise RuntimeError('Pre-spawn original reserves plus full planned worker RSS')
    child = subprocess.Popen([sys.executable,'-B','-X','pycache_prefix='+str(out/'_fresh_no_write_cache'),
        str(ROOT/SCRIPT),'--worker','--preregistration-commit',args.preregistration_commit,'--git-proxy',args.git_proxy],
        cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    try:
        while child.poll() is None:
            p = Path(f'/proc/{child.pid}/status')
            try:
                status = p.read_text()
                peak = max(peak,prior.runtime.process_memory(status,Path(f'/proc/{child.pid}/smaps_rollup').read_text())['guarded_process_RSS_bytes'])
                fields = dict(l.split(':',1) for l in status.splitlines() if ':' in l); threads = max(threads,int(fields['Threads']))
            except (FileNotFoundError,ProcessLookupError,RuntimeError):
                if child.poll() is not None: break
                try: settled = prior.runtime.kernel_terminal_status(p.read_text())
                except (FileNotFoundError,ProcessLookupError): settled = False
                if settled: child.wait(timeout=1); break
                try: child.wait(timeout=.1); break
                except subprocess.TimeoutExpired: pass
                if time.monotonic()-start>2: reason='owned_memory_counter_unavailable'; child.kill(); break
                time.sleep(.2); continue
            reason = prior.runtime.resources_ok(cfg,safety,ROOT,start,peak)
            if prior.runtime.output_bytes(out)>cfg['output_bytes_ceiling']: reason='owned_output_size'
            try: helpers = max(helpers,prior.runtime.inspect_owned_threads(child.pid))
            except ValueError: reason='owned_worker_threads'
            except (FileNotFoundError,ProcessLookupError):
                if child.poll() is not None: break
                reason='owned_thread_inventory_unavailable'
            if reason:
                if child.poll() is None: child.kill()  # ONLY this freshly-created owned Popen child.
                break
            time.sleep(.2)
        code = child.wait(timeout=10)
    except BaseException:
        if child.poll() is None: child.kill()
        child.wait(timeout=10); raise
    complete = (out/'result.json').is_file()
    atomic_json(terminal,{'owned_worker_exit_code':code,'guard_stop_reason':reason,
        'successful_owned_execution':code==0 and reason is None and complete,'complete_result_present':complete,
        'peak_sampled_owned_worker_RSS_bytes':peak,'peak_sampled_worker_threads':threads,
        'peak_identified_native_driver_helpers':helpers,'CPU_compute_workers':1,'training_GPU_hours':0,
        'only_new_owned_Popen_worker_managed':True,'wall_seconds_including_worker_imports_network':time.monotonic()-start})
    return 0 if code==0 and reason is None and complete else 1


def save_own_checkpoint(out,control,model,guard,*,optimizer=None,step=None,rng=None,solver=None):
    """Exact own model+AdamW/RNG reopen; solver record is NOT resume state."""
    import torch
    from src.trackocd_cvpr.raw_known_membership_control import NamedKnownLogit,parameter_sha256
    from src.trackocd_cvpr.fresh_A_training import checkpoint_state_equal
    if control not in ('calendar_matched','full_policy') or not out.is_dir():
        raise ValueError('Only own fixed dual-control output checkpoints')
    path = out/f'private_{control}_pilot.pt'; tmp = path.with_suffix('.partial')
    if path.exists() or tmp.exists(): raise ValueError('Never overwrite an own existing checkpoint')
    payload = {'control':control,'model_state':model.state_dict(),'step':step,
        'optimizer_state':None if optimizer is None else optimizer.state_dict(),
        'numpy_rng_state':rng,'torch_rng_state':torch.get_rng_state(),
        'solver_termination_record_not_internal_resume_state':solver}
    with tmp.open('xb') as f:
        os.chmod(tmp,0o600); torch.save(payload,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path); guard(); restored = torch.load(path,map_location='cpu',weights_only=True)
    verifier = NamedKnownLogit(); verifier.load_state_dict(restored['model_state'],strict=True)
    if (parameter_sha256(verifier)!=parameter_sha256(model)
            or not checkpoint_state_equal(payload,restored)):
        raise ValueError('Actual exact own model/optimizer/RNG or solver-record reopen')
    return {'control':control,'step':step,'sha256':sha256_file(path),'bytes':path.stat().st_size,
        'reopen_exact':True,'AdamW_and_RNG_reopened':optimizer is not None,
        'solver_internal_LBFGS_resume_state_claimed':False}


def worker(args,cfg,safety):
    out = ROOT/OUT
    if (not sys.dont_write_bytecode or sys.pycache_prefix!=str(out/'_fresh_no_write_cache')
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=''):
        raise ValueError('Fresh hidden-GPU no-write CPU pilot worker')
    proof = source_gate(args,cfg,safety)
    import numpy as np
    import torch
    import scipy
    from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
    from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks
    from src.trackocd_cvpr.R_development_evaluation import known_prototypes
    from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
    from src.trackocd_cvpr.spread_known_calendar import freeze_calendar
    from src.trackocd_cvpr.raw_known_membership_control import (
        NamedKnownLogit,parameter_sha256,collect_full_policy_samples,objective_and_gradient,INITIAL,
        membership_diagnostics,calendar_matched_fit,full_policy_fit)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    def barrier(event,values):
        if event=='socket.connect': raise PermissionError('No network after registration')
        if event=='open' and isinstance(values[0],(str,bytes)):
            p = Path(os.fsdecode(values[0])).resolve()
            if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (p.suffix in ('.pt','.pth','.partial') and p.parent!=out):
                raise PermissionError('Qualified caches and only own fresh pilot checkpoints, no original pixels/GT/oldweights')
    sys.addaudithook(barrier)
    if out.exists(): raise ValueError('No output overwrite')
    out.mkdir(parents=True); start = time.monotonic(); peak = 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT/'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak,prior.runtime.process_memory()['guarded_process_RSS_bytes'])
        size = prior.runtime.output_bytes(out); reason = prior.runtime.resources_ok(cfg,safety,ROOT,start,peak)
        if reason or size>cfg['output_bytes_ceiling'] or prior_bytes+size>safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned dual-control pilot guard:'+str(reason or 'output_storage'))
        if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1 or torch.cuda.is_initialized():
            raise RuntimeError('One CPU compute worker/no CUDA')
    atomic_json(out/'launch.json',{'preregistration_commit':args.preregistration_commit,
        'live_remote_registration_proof':proof,'source_sha256':{p:sha256_file(ROOT/p) for p in SOURCES},
        'scope':'One explicit lawful-policy dual pilot: calendar50 and deterministic fullpolicy solver<=50, no formal/dev auto-start'})
    try:
        guard(); videos,projection,policy = load_completed_policy_videos(ROOT,cfg,guard)
        fit,fit_scope = load_completed_fit_tracks(ROOT,cfg,guard)
        raw,_ = known_prototypes(RawPreservingEvidence().eval().requires_grad_(False),fit)
        known = {i:v for i,v in enumerate(raw)}; pilot = cfg['pilot']; saved = []
        private,collection = collect_full_policy_samples(videos,projection,known,guard)
        if (len(videos),collection['physical_observations'],collection['metadata_frames'],collection['empty_frames'])!=(10,1268,374,1):
            raise ValueError('Complete actual lawful policy clock incl masked groups')
        initial = NamedKnownLogit().eval().requires_grad_(False)
        if not all((float(initial(float(x)))>=0)==(float(x)>=.65) for x in private[:,0]):
            raise ValueError('Actual initial NamedKnown scalar gate changed inherited Raw threshold')
        before = membership_diagnostics(initial,private)
        frozen = freeze_calendar(videos,projection,pilot['seed'],pilot['episodes'],pilot['calendar_updates'],guard)
        atomic_json(out/'private_frozen_calendar.json',frozen); os.chmod(out/'private_frozen_calendar.json',0o600)
        def checkpoint(model,optimizer,step,rng):
            if step!=50: raise ValueError('Only fixed calendar50 pilot snapshot')
            saved.append(save_own_checkpoint(out,'calendar_matched',model,guard,optimizer=optimizer,step=step,rng=rng))
        calendar,_,calendar_summary,_ = calendar_matched_fit(videos,projection,known,pilot['seed'],
            frozen_calendar=frozen,episodes=pilot['episodes'],updates_per_episode=pilot['calendar_updates'],
            checkpoint_steps=pilot['calendar_checkpoints'],guard=guard,
            progress=lambda row:atomic_json(out/'progress.json',{'calendar_last_step':row}),checkpoint=checkpoint)
        full_initial_loss,full_initial_grad = objective_and_gradient(INITIAL,private)
        full,full_summary = full_policy_fit(private,iteration_budget=pilot['full_policy_maxiter'],
            function_budget=pilot['full_policy_maxfun'],guard=guard)
        saved.append(save_own_checkpoint(out,'full_policy',full,guard,step=full_summary['iterations'],solver=full_summary))
        gradient = float(np.linalg.norm(full_initial_grad))
        both_roles_updated = all(sum(r['lawful_group_counts'].get(role,0) for r in calendar_summary['loss_curve'])>0
                                 for role in ('Known','policy_pseudo'))
        operational = bool(full_summary['solver_success'] and gradient>0 and both_roles_updated
                           and parameter_sha256(full)!=parameter_sha256(initial) and len(saved)==2)
        guard()
        result = {'status':'COMPLETED_FIXED_RAW_MEMBERSHIP_DUAL_PILOT_NOT_DEV_CALIBRATION_FOREGROUND_OR_GOAL',
            'live_remote_registration_proof':proof,'policy_scope':policy,'offline_fit_prior_scope':fit_scope,
            'full_policy_collection':collection,'initial_NamedKnown_gate_exact_on_actual_scores':True,
            'calendar_control':calendar_summary,'full_policy_control':{**full_summary,
                'initial_regularized_class_macro_NLL':full_initial_loss,'initial_gradient_norm':gradient},
            'resubstitution_diagnostics_on_same_training_samples':{'initial':before,
                'calendar_matched':membership_diagnostics(calendar,private),'full_policy':membership_diagnostics(full,private)},
            'private_checkpoints':saved,'operational_prerequisites_for_potential_later_registration':operational,
            'actual_calibration_generalization_or_control_winner_selected':False,
            'pilot_warmstart_or_automatic_formal_dev_start':False,'dev_final_Val_Test_or_new_nonKnown_inputs_used':False,
            'raw_private_GT_scalar_table_weights_or_traces_exported':False,
            'Existing_gate_foreground_purity_or_deployment_posterior_calibrated':False,
            'libraries':{'python':sys.version.split()[0],'torch':torch.__version__,'scipy':scipy.__version__},
            'resources':{'peak_guarded_RSS_bytes':peak,'wall_seconds_excluding_network':time.monotonic()-start,
                         'training_GPU_hours':0,'CUDA_initialized':torch.cuda.is_initialized()}}
        atomic_json(out/'result.json',result)
        print(json.dumps({'status':result['status'],'operational_prerequisites':operational,
                          'solver_success':full_summary['solver_success']}),flush=True)
    except Exception as error:
        atomic_json(out/'failure.json',{'exception_type':type(error).__name__,'error':str(error),'complete':False}); raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit',required=True)
    p.add_argument('--git-proxy',default=''); p.add_argument('--worker',action='store_true'); args = p.parse_args()
    cfg = configuration(ROOT); safety = json.loads((ROOT/'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker: worker(args,cfg,safety); return 0
    return supervise(args,cfg,safety)


if __name__=='__main__': sys.exit(main())
