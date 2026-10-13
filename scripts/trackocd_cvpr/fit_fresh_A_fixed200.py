#!/usr/bin/env python3
"""One explicit gated fresh fit5 Fixed200 CPU trial, not automatic all15/dev."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration
from src.trackocd_cvpr.fresh_A_formal_fit import CONFIG,configuration,verify_pilot,active_research_workers,PILOT_SOURCES
from scripts.trackocd_cvpr.pilot_fresh_A_optimizer import SOURCES as BASE_SOURCES,output_bytes
from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok,kernel_terminal_status
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory
from src.trackocd_cvpr.fresh_A_CPU_recovery import inspect_owned_threads

SCRIPT='scripts/trackocd_cvpr/fit_fresh_A_fixed200.py'
SOURCES=tuple(dict.fromkeys((*BASE_SOURCES,*PILOT_SOURCES,CONFIG,SCRIPT,'src/trackocd_cvpr/fresh_A_formal_fit.py')))


def source_gate(args,cfg,safety):
    proof=verify_registration(ROOT,safety['branch'],args.preregistration_commit,args.git_proxy,allow_public_api=True)
    for p in SOURCES:
        if (ROOT/p).is_symlink() or subprocess.check_output(['git','show',args.preregistration_commit+':'+p],cwd=ROOT)!=(ROOT/p).read_bytes():
            raise ValueError('Exact source/pilot/cache/authority registration before any private input')
    authority=json.loads((ROOT/cfg['approval']).read_text())
    if authority['status']!='USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN':raise ValueError('Explicit bounded training authority')
    return proof,verify_pilot(ROOT,cfg)


def supervise(args,cfg,safety):
    out=ROOT/cfg['output_directory']; terminal=out.with_name(out.name+'_supervisor_attempt_1.json')
    if out.exists() or terminal.exists():raise ValueError('Preserve every existing trial/terminal')
    if active_research_workers(ROOT):raise RuntimeError('Owned M evaluation/fresh fit active; observe, do not start another fit')
    source_gate(args,cfg,safety)
    started=time.monotonic();peak=0;threads=0;helpers=0;reason=None
    if resources_ok(cfg,safety,ROOT,started,0):raise RuntimeError('Pre-spawn resource guard')
    child=subprocess.Popen([sys.executable,'-B','-X','pycache_prefix='+str(out/'_fresh_no_write_cache'),
        str(ROOT/SCRIPT),'--worker','--family',args.family,'--seed',str(args.seed),
        '--preregistration-commit',args.preregistration_commit,'--git-proxy',args.git_proxy],cwd=ROOT,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    try:
        while child.poll() is None:
            status_path=Path(f'/proc/{child.pid}/status')
            try:
                status=status_path.read_text();rollup=Path(f'/proc/{child.pid}/smaps_rollup').read_text()
                peak=max(peak,process_memory(status,rollup)['guarded_process_RSS_bytes'])
                fields=dict(l.split(':',1) for l in status.splitlines() if ':' in l);threads=max(threads,int(fields['Threads'].strip()))
            except (FileNotFoundError,ProcessLookupError,RuntimeError):
                if child.poll() is not None:break
                try: settled=kernel_terminal_status(status_path.read_text())
                except (FileNotFoundError,ProcessLookupError):settled=False
                if settled:child.wait(timeout=1);break
                if time.monotonic()-started>2:reason='owned_memory_counter_unavailable';child.kill();break
                time.sleep(.2);continue
            reason=resources_ok(cfg,safety,ROOT,started,peak)
            if output_bytes(out)>cfg['output_bytes_ceiling']:reason='owned_output_size'
            try:helpers=max(helpers,inspect_owned_threads(child.pid))
            except ValueError:reason='owned_worker_threads'
            except (FileNotFoundError,ProcessLookupError):
                if child.poll() is not None:break
                reason='owned_thread_inventory_unavailable'
            if reason:
                if child.poll() is None:child.kill()  # ONLY this newly created Popen child
                break
            time.sleep(.2)
        code=child.wait(timeout=10)
    except BaseException:
        if child.poll() is None:child.kill()
        child.wait(timeout=10);raise
    if code==0 and reason is None and not (out/'result.json').is_file():reason='owned_result_missing'
    steps=json.loads((out/'progress.json').read_text())['step'] if (out/'progress.json').is_file() else 0
    atomic_json(terminal,{'owned_worker_exit_code':code,'guard_stop_reason':reason,
        'successful_owned_execution':code==0 and reason is None,'full_result_present':(out/'result.json').is_file(),
        'last_persisted_completed_optimizer_steps':steps,'only_new_owned_Popen_worker_managed':True,
        'peak_sampled_owned_worker_RSS_bytes':peak,'peak_sampled_worker_threads':threads,
        'peak_identified_native_driver_helpers':helpers,'CPU_compute_workers':1,'training_GPU_hours':0,
        'execution_device':'CPU','wall_seconds_including_worker_imports_and_network':time.monotonic()-started})
    return 0 if code==0 and reason is None else 1


def worker(args,cfg,safety):
    out=ROOT/cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix!=str(out/'_fresh_no_write_cache') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Fresh no-write, hidden GPU owned CPU worker')
    proof,pilot=source_gate(args,cfg,safety)
    import torch
    from src.trackocd_cvpr.fresh_A_training import validate_protocol,fit_baseline,checkpoint_state_equal
    from src.trackocd_cvpr.fresh_A_baseline import fresh_model
    from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks
    from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
    protocol=json.loads((ROOT/cfg['protocol']).read_text());validate_protocol(protocol)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if out.exists():raise ValueError('No old checkpoint/result overwrite')
    def barrier(event,values):
        if event=='socket.connect':raise PermissionError('No network after source gate')
        if event=='open' and isinstance(values[0],(str,bytes)):
            p=Path(os.fsdecode(values[0])).resolve()
            if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')):raise PermissionError('Cache-only fit, no original GT/pixels')
    sys.addaudithook(barrier)
    out.mkdir(parents=True);started=time.monotonic();peak=0
    prior=sum(p.stat().st_size for p in (ROOT/'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak=max(peak,process_memory()['guarded_process_RSS_bytes']);size=output_bytes(out)
        reason=resources_ok(cfg,safety,ROOT,started,peak)
        if reason or size>cfg['output_bytes_ceiling'] or prior+size>safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned Fixed200 guard:'+str(reason or 'output_storage'))
        if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1 or torch.cuda.is_initialized():
            raise RuntimeError('One CPU compute worker, no CUDA tensors')
    atomic_json(out/'launch.json',{'preregistration_commit':args.preregistration_commit,
        'live_remote_registration_proof':proof,'source_sha256':{p:sha256_file(ROOT/p) for p in SOURCES},
        'pilot_prerequisite':pilot,'scope':'One fresh fit5 Fixed200 baseline, no dev or automatic15fits'})
    try:
        guard();data_cfg=json.loads((ROOT/cfg['data_config']).read_text())
        tracks,label_proof=load_completed_fit_tracks(ROOT,data_cfg,guard);checkpoints=[]
        def progress(row,curve):atomic_json(out/'progress.json',row);atomic_json(out/'loss_curve.json',curve)
        def save(model,optimizer,step,metadata):
            path=out/f'private_step{step}_checkpoint.pt';tmp=path.with_suffix('.pt.partial')
            with tmp.open('xb') as f:
                os.chmod(tmp,0o600);torch.save({'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                    'step':step,**metadata},f);f.flush();os.fsync(f.fileno())
            if path.exists():raise ValueError('Preserve fixed checkpoint')
            os.replace(tmp,path);guard();restored=torch.load(path,map_location='cpu',weights_only=True)
            verifier=fresh_model(args.family,args.seed);verifier.load_state_dict(restored['model_state'],strict=True)
            if (digest(verifier)!=digest(model) or restored['step']!=step or restored['family']!=args.family
                    or restored['seed']!=args.seed or restored['stage']!='Fixed200'
                    or not checkpoint_state_equal(restored['optimizer_state'],optimizer.state_dict())):
                raise ValueError('Actual model/AdamW fixed checkpoint reopen differs')
            checkpoints.append({'step':step,'bytes':path.stat().st_size,'sha256':sha256_file(path),'reopen_exact':True})
        model,optimizer,trial=fit_baseline(tracks,protocol,args.family,args.seed,'Fixed200',guard,progress,save)
        if [r['step'] for r in checkpoints]!=[20,50,100,200] or trial['scope']['physical_tracks']!=1401:
            raise ValueError('All fixed snapshots/full physical census')
        result={'status':'COMPLETED_ONE_FRESH_FIXED200_FIT_NOT_ALL15_DEV_OR_GOAL_ACCEPTANCE',
            'live_remote_registration_proof':proof,'pilot_prerequisite':pilot,'trial':trial,
            'label_and_physical_scope':label_proof,'private_checkpoints':checkpoints,
            'checkpoint_selection':'Fixed200, never dev/final/Val/Test selection',
            'pilot_or_historical_checkpoint_warmstart':False,'dev_final_Val_Test_inputs_used':False,
            'formal15fits_dev_or_Goal_accepted':False,'resources':{'peak_guarded_RSS_bytes':peak,
                'wall_seconds_excluding_network':time.monotonic()-started,'training_GPU_hours':0,
                'CUDA_initialized':torch.cuda.is_initialized()}}
        guard()
        if output_bytes(out)+len(json.dumps(result,indent=2).encode())+4096>cfg['output_bytes_ceiling']:raise RuntimeError('Final result output ceiling')
        atomic_json(out/'result.json',result);print(json.dumps({'status':result['status'],'family':args.family,'seed':args.seed}),flush=True)
    except Exception as e:
        atomic_json(out/'failure.json',{'exception_type':type(e).__name__,'error':str(e),'complete':False});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit',required=True);p.add_argument('--git-proxy',default='')
    p.add_argument('--family',required=True,choices=('A1_ADAPTER','A1_GEOMETRY_1','A1_GEOMETRY_5','A2_EVIDENCE','A2_CAPACITY_CONTROL'))
    p.add_argument('--seed',required=True,type=int,choices=(1027,1028,1029));p.add_argument('--worker',action='store_true');args=p.parse_args()
    cfg=configuration(ROOT,args.family,args.seed);safety=json.loads((ROOT/'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker:worker(args,cfg,safety);return 0
    return supervise(args,cfg,safety)


if __name__=='__main__':sys.exit(main())
