#!/usr/bin/env python3
"""One explicitly selected fresh scalar-control fit; fixed snapshots/no dev."""
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
from src.trackocd_cvpr.raw_membership_formal_protocol import configuration as formal_configuration,CONFIG
from src.trackocd_cvpr.raw_membership_pilot_qualification import HASHES
from scripts.trackocd_cvpr import pilot_raw_known_membership_controls as pilot

SCRIPT = 'scripts/trackocd_cvpr/fit_raw_known_membership_formal.py'
SOURCES = tuple(dict.fromkeys((*pilot.SOURCES,CONFIG,SCRIPT,*HASHES,
    'src/trackocd_cvpr/raw_membership_formal_protocol.py','src/trackocd_cvpr/raw_membership_pilot_qualification.py')))


def configuration(root,args):
    return {**pilot.configuration(root),**formal_configuration(root,args.control,args.seed)}


def source_gate(args,cfg,safety):
    if cfg!=configuration(ROOT,args): raise ValueError('Exact formal invocation before private input')
    proof = verify_registration(ROOT,safety['branch'],args.preregistration_commit,args.git_proxy,allow_public_api=True)
    for p in SOURCES:
        if (ROOT/p).is_symlink() or subprocess.check_output(['git','show',args.preregistration_commit+':'+p],cwd=ROOT)!=(ROOT/p).read_bytes():
            raise ValueError('All qualified pilot/source/public evidence actually registered')
    authority = json.loads((ROOT/cfg['approval']).read_text())
    if (authority['status']!='USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or authority['video_scope']!=407 or authority['allowed_training_label_roles']!=['fit_known','policy_train_pseudo']
            or authority['new_nonKnown_semantic_GT_or_Novel_vocabulary_authorized']
            or authority['unlabeled_objects_background_or_inferred_Novel_labels']
            or authority['old_51_heldout_Val_Test_DEVplus_Q1_public_new_sealed_and_other_videos_allowed']
            or authority['foreign_process_interference_allowed']):
        raise ValueError('Actual human-approved same fit/policy roles')
    q = ROOT/'outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1'
    t = json.loads(q.with_name(q.name+'_supervisor_attempt_1.json').read_text())
    if (sha256_file(q/'result.json')!=cfg['policy_input_qualification_sha256']
            or not t['successful_owned_execution'] or t['owned_worker_exit_code']!=0 or t['guard_stop_reason'] is not None):
        raise ValueError('Actual qualified complete policy input')
    return proof


def busy():
    if pilot.busy(): return True
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name)==os.getpid(): continue
        try:
            words=p.read_bytes().split(b'\0'); state=(p.parent/'stat').read_text().rsplit(')',1)[1].split()[0]
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        if os.fsencode(str(ROOT/SCRIPT)) in words and state not in ('Z','X','x'): return True
    return False


def supervise(args,cfg,safety):
    out = ROOT/cfg['output_directory']; terminal=out.with_name(out.name+'_supervisor_attempt_1.json')
    if out.exists() or terminal.exists(): raise ValueError('One fresh formal control attempt, preserve every prior output')
    if busy(): raise RuntimeError('Observe live owned research job; no parallel scalar fit')
    source_gate(args,cfg,safety); start=time.monotonic(); peak=threads=helpers=0; reason=None
    mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    if (pilot.prior.runtime.resources_ok(cfg,safety,ROOT,start,0)
            or mem['MemAvailable']<safety['system_ram_reserve_fraction']*mem['MemTotal']+cfg['rss_bytes_ceiling']):
        raise RuntimeError('Pre-spawn original resource reserve plus planned RSS')
    extra=['--control',args.control]+(['--seed',str(args.seed)] if args.seed is not None else [])
    child=subprocess.Popen([sys.executable,'-B','-X','pycache_prefix='+str(out/'_fresh_no_write_cache'),
        str(ROOT/SCRIPT),'--worker','--preregistration-commit',args.preregistration_commit,'--git-proxy',args.git_proxy,*extra],
        cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    try:
        while child.poll() is None:
            p=Path(f'/proc/{child.pid}/status')
            try:
                status=p.read_text(); peak=max(peak,pilot.prior.runtime.process_memory(status,Path(f'/proc/{child.pid}/smaps_rollup').read_text())['guarded_process_RSS_bytes'])
                fields=dict(l.split(':',1) for l in status.splitlines() if ':' in l); threads=max(threads,int(fields['Threads']))
            except (FileNotFoundError,ProcessLookupError,RuntimeError):
                if child.poll() is not None: break
                try: settled=pilot.prior.runtime.kernel_terminal_status(p.read_text())
                except (FileNotFoundError,ProcessLookupError): settled=False
                if settled: child.wait(timeout=1); break
                try: child.wait(timeout=.1); break
                except subprocess.TimeoutExpired: pass
                if time.monotonic()-start>2: reason='owned_memory_counter_unavailable'; child.kill(); break
                time.sleep(.2); continue
            reason=pilot.prior.runtime.resources_ok(cfg,safety,ROOT,start,peak)
            if pilot.prior.runtime.output_bytes(out)>cfg['output_bytes_ceiling']: reason='owned_output_size'
            try: helpers=max(helpers,pilot.prior.runtime.inspect_owned_threads(child.pid))
            except ValueError: reason='owned_worker_threads'
            except (FileNotFoundError,ProcessLookupError):
                if child.poll() is not None: break
                reason='owned_thread_inventory_unavailable'
            if reason:
                if child.poll() is None: child.kill()  # Only this new owned Popen child.
                break
            time.sleep(.2)
        code=child.wait(timeout=10)
    except BaseException:
        if child.poll() is None: child.kill()
        child.wait(timeout=10); raise
    complete=(out/'result.json').is_file()
    atomic_json(terminal,{'owned_worker_exit_code':code,'guard_stop_reason':reason,
        'successful_owned_execution':code==0 and reason is None and complete,'complete_result_present':complete,
        'peak_sampled_owned_worker_RSS_bytes':peak,'peak_sampled_worker_threads':threads,
        'peak_identified_native_driver_helpers':helpers,'CPU_compute_workers':1,'training_GPU_hours':0,
        'only_new_owned_Popen_worker_managed':True,'wall_seconds_including_worker_imports_network':time.monotonic()-start})
    return 0 if code==0 and reason is None and complete else 1


def save(out,model,guard,control,step,*,optimizer=None,rng=None,solver=None):
    import torch
    from src.trackocd_cvpr.raw_known_membership_control import NamedKnownLogit,parameter_sha256
    from src.trackocd_cvpr.fresh_A_training import checkpoint_state_equal
    if (control not in ('calendar_matched','full_policy') or not out.is_dir() or type(step) is not int
            or control=='calendar_matched' and step not in (20,50,100,200)
            or control=='full_policy' and not 0<=step<=200):
        raise ValueError('Only own fixed formal snapshot steps/control')
    path=out/(f'private_step{step}_checkpoint.pt' if control=='calendar_matched' else 'private_full_policy_checkpoint.pt')
    tmp=path.with_suffix('.partial')
    if path.exists() or tmp.exists(): raise ValueError('Preserve every own fixed snapshot')
    payload={'control':control,'step':step,'model_state':model.state_dict(),
        'optimizer_state':None if optimizer is None else optimizer.state_dict(),
        'numpy_rng_state':rng,'torch_rng_state':torch.get_rng_state(),'solver_termination_record_not_resume_state':solver}
    with tmp.open('xb') as f:
        os.chmod(tmp,0o600); torch.save(payload,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path); guard(); restored=torch.load(path,map_location='cpu',weights_only=True)
    verifier=NamedKnownLogit(); verifier.load_state_dict(restored['model_state'],strict=True)
    if parameter_sha256(verifier)!=parameter_sha256(model) or not checkpoint_state_equal(payload,restored):
        raise ValueError('Exact actual own model/AdamW/RNG or solver record reopening')
    return {'control':control,'step':step,'sha256':sha256_file(path),'bytes':path.stat().st_size,
        'reopen_exact':True,'AdamW_and_RNG_reopened':optimizer is not None,'solver_internal_resume_state_claimed':False}


def worker(args,cfg,safety):
    out=ROOT/cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix!=str(out/'_fresh_no_write_cache') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Fresh hidden-GPU no-write CPU worker')
    proof=source_gate(args,cfg,safety)
    import numpy as np
    import torch
    from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
    from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks
    from src.trackocd_cvpr.R_development_evaluation import known_prototypes
    from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
    from src.trackocd_cvpr.spread_known_calendar import freeze_calendar
    from src.trackocd_cvpr.raw_known_membership_control import (
        NamedKnownLogit,parameter_sha256,collect_full_policy_samples,membership_diagnostics,calendar_matched_fit,
        full_policy_fit,objective_and_gradient,INITIAL)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    def barrier(event,values):
        if event=='socket.connect': raise PermissionError('No network after registration')
        if event=='open' and isinstance(values[0],(str,bytes)):
            p=Path(os.fsdecode(values[0])).resolve()
            if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (p.suffix in ('.pt','.pth','.partial') and p.parent!=out):
                raise PermissionError('Qualified caches and own fresh formal snapshots only; no pilot/old/pixel/annotation reads')
    sys.addaudithook(barrier)
    if out.exists(): raise ValueError('No prior output overwrite')
    out.mkdir(parents=True); start=time.monotonic(); peak=0
    prior_bytes=sum(p.stat().st_size for p in (ROOT/'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak=max(peak,pilot.prior.runtime.process_memory()['guarded_process_RSS_bytes'])
        size=pilot.prior.runtime.output_bytes(out); reason=pilot.prior.runtime.resources_ok(cfg,safety,ROOT,start,peak)
        if reason or size>cfg['output_bytes_ceiling'] or prior_bytes+size>safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned fresh formal scalar-control guard:'+str(reason or 'output_storage'))
        if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1 or torch.cuda.is_initialized():
            raise RuntimeError('One CPU compute worker/no CUDA')
    atomic_json(out/'launch.json',{'preregistration_commit':args.preregistration_commit,'live_remote_registration_proof':proof,
        'source_sha256':{p:sha256_file(ROOT/p) for p in SOURCES},'control':args.control,'calendar_seed':args.seed,
        'scope':'One fresh fixed formal control, no pilot warmstart/model selection/dev auto-start'})
    try:
        guard(); videos,projection,policy=load_completed_policy_videos(ROOT,cfg,guard)
        fit,fit_scope=load_completed_fit_tracks(ROOT,cfg,guard)
        raw,_=known_prototypes(RawPreservingEvidence().eval().requires_grad_(False),fit)
        known={i:v for i,v in enumerate(raw)}; private,collection=collect_full_policy_samples(videos,projection,known,guard)
        if (len(videos),collection['physical_observations'],collection['metadata_frames'],collection['empty_frames'])!=(10,1268,374,1):
            raise ValueError('Complete unchanged policy physical universe')
        initial=NamedKnownLogit().eval().requires_grad_(False); before=membership_diagnostics(initial,private); saved=[]
        if args.control=='calendar_matched':
            frozen=freeze_calendar(videos,projection,args.seed,4,50,guard)
            atomic_json(out/'private_frozen_calendar.json',frozen); os.chmod(out/'private_frozen_calendar.json',0o600)
            def checkpoint(model,optimizer,step,rng):
                saved.append(save(out,model,guard,args.control,step,optimizer=optimizer,rng=rng))
            model,_,summary,_=calendar_matched_fit(videos,projection,known,args.seed,frozen_calendar=frozen,
                episodes=4,updates_per_episode=50,checkpoint_steps=cfg['calendar_checkpoint_steps'],guard=guard,
                progress=lambda row:atomic_json(out/'progress.json',row),checkpoint=checkpoint)
            if summary['optimizer_steps']!=200 or [s['step'] for s in saved]!=[20,50,100,200]:
                raise ValueError('All200realupdates and4snapshots required')
            if any((e['physical_observations'],e['metadata_frames'],e['empty_frames'])!=(1268,374,1) for e in summary['episodes']):
                raise ValueError('Every full episode clock preserved')
            old=ROOT/f'outputs/trackocd_cvpr/cvpr3/paired_spread_Known_rejection_Fixed200_v1_s{args.seed}/result.json'
            if summary['actual_shared_calendar_sha256']!=json.loads(old.read_text())['summary']['actual_shared_physical_calendar_sha256']:
                raise ValueError('Actual matched C physical update calendar, not config assertion')
            solver_converged=None
        else:
            _,gradient=objective_and_gradient(INITIAL,private)
            model,summary=full_policy_fit(private,iteration_budget=cfg['full_policy_maxiter'],function_budget=cfg['full_policy_maxfun'],guard=guard)
            summary['initial_gradient_norm']=float(np.linalg.norm(gradient))
            saved.append(save(out,model,guard,args.control,summary['iterations'],solver=summary))
            solver_converged=summary['solver_success']
        if parameter_sha256(model)==parameter_sha256(initial): raise ValueError('Fresh actual fit, not unchanged parameters')
        guard(); result={'status':'COMPLETED_FRESH_FORMAL_RAW_MEMBERSHIP_CONTROL_NOT_DEV_CALIBRATION_OR_GOAL',
            'live_remote_registration_proof':proof,'control':args.control,'calendar_seed':args.seed,
            'policy_scope':policy,'offline_fit_prior_scope':fit_scope,'full_policy_collection':collection,'summary':summary,
            'private_checkpoints':saved,'solver_converged':solver_converged,
            'resubstitution_diagnostics_on_same_training_samples':{'initial':before,'fixed_formal_endpoint':membership_diagnostics(model,private)},
            'model_parameter_sha256':parameter_sha256(model),'initialization_sha256':parameter_sha256(initial),
            'old_or_pilot_weight_warmstart':False,'dev_final_Val_Test_or_new_nonKnown_inputs_used':False,
            'solver_resume_state_transfer_calibration_or_control_winner_claimed':False,
            'all4formal_fits_dev_or_Goal_accepted':False,
            'resources':{'peak_guarded_RSS_bytes':peak,'wall_seconds_excluding_network':time.monotonic()-start,
                         'training_GPU_hours':0,'CUDA_initialized':torch.cuda.is_initialized()}}
        atomic_json(out/'result.json',result)
        print(json.dumps({'status':result['status'],'control':args.control,'calendar_seed':args.seed,'solver_converged':solver_converged}),flush=True)
    except Exception as error:
        atomic_json(out/'failure.json',{'exception_type':type(error).__name__,'error':str(error),'complete':False}); raise


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preregistration-commit',required=True); p.add_argument('--git-proxy',default='')
    p.add_argument('--control',choices=('calendar_matched','full_policy'),required=True)
    p.add_argument('--seed',type=int,choices=(1027,1028,1029)); p.add_argument('--worker',action='store_true'); args=p.parse_args()
    cfg=configuration(ROOT,args); safety=json.loads((ROOT/'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker: worker(args,cfg,safety); return 0
    return supervise(args,cfg,safety)


if __name__=='__main__': sys.exit(main())
