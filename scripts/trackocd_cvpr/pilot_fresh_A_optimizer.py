#!/usr/bin/env python3
"""Owned single-CPU five-family pilot; no formal fits or dev auto-launch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration
from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok,kernel_terminal_status
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory
from scripts.trackocd_cvpr.smoke_fit_R_optimizer import SOURCES as FIT_SOURCES
from src.trackocd_cvpr.fresh_A_CPU_recovery import recovery_configuration,verify_original,inspect_owned_threads,RETRY,SOURCES as RECOVERY_SOURCES

CONFIG='configs/trackocd_cvpr/fresh_A_optimizer_pilot_v1.json'
PROTOCOL='configs/trackocd_cvpr/fresh_A_fit5_baselines_v1.json'
SOURCES=tuple(dict.fromkeys((*FIT_SOURCES,CONFIG,PROTOCOL,
    'scripts/trackocd_cvpr/pilot_fresh_A_optimizer.py','src/trackocd_cvpr/fresh_A_baseline.py',
    'src/trackocd_cvpr/fresh_A_training.py','src/trackocd_core/representation.py',
    'src/trackocd_core/train_first_representation.py','src/trackocd_cvpr/differentiable_support_bank.py',
    'src/trackocd_cvpr/train_physical_resume.py','src/trackocd_cvpr/joint_known_protocol.py',
    'src/trackocd_cvpr/known_census_protocol.py','src/trackocd_cvpr/global_protocol_coverage.py',
    'src/trackocd_cvpr/physical_gt_census.py','src/trackocd_cvpr/selected_image_inventory.py',
    'scripts/trackocd_cvpr/audit_known_census_protocol.py','src/trackocd_cvpr/fresh_A_CPU_recovery.py',
    'configs/trackocd_cvpr/fresh_A_cpu_visibility_retry1.json')))


def validate_execution(cfg,authority):
    namespace=(RETRY if cfg.get('CPU_visibility_recovery')=='cpu_visibility_retry1'
               else 'outputs/trackocd_cvpr/cvpr2/fresh_A_fit5_optimizer_pilot_v1')
    if (authority['status']!='USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or cfg['status']!='REGISTERED_FRESH_FIT5_FIVE_FAMILY_PILOT_ACTUAL_EXECUTION_PENDING'
            or cfg['protocol']!=PROTOCOL or cfg['data_config']!='configs/trackocd_cvpr/fit_R_optimizer_smoke_v1.json'
            or cfg['approval']!='configs/trackocd_cvpr/formal_research_scope_approval_v1.json'
            or cfg['execution_device']!='CPU' or cfg['CPU_compute_workers']!=1
            or cfg['rss_bytes_ceiling']!=3221225472 or cfg['wall_seconds_ceiling']!=3000
            or cfg['output_bytes_ceiling']!=33554432
            or cfg['output_directory']!=namespace
            or cfg['full_scientific_fit_automatic_start'] or cfg['formal15fits_development_or_Goal_accepted']):
        raise ValueError('Exact bounded five-family fit-only pilot authority/configuration')


def output_bytes(out):
    # Count partial checkpoint files too, never follow an injected symlink.
    entries=list(out.rglob('*')) if out.exists() else []
    if any(p.is_symlink() for p in entries): raise ValueError('No output symlinks')
    return sum(p.stat().st_size for p in entries if p.is_file())


def supervise(args,cfg,safety):
    out=ROOT/cfg['output_directory']; receipt=out.with_name(out.name+'_supervisor_attempt_1.json')
    if out.exists() or receipt.exists(): raise ValueError('Preserve original namespace/terminal')
    started=time.monotonic(); peak=0; threads=0; driver_helpers=0; reason=None
    if resources_ok(cfg,safety,ROOT,started,0): raise RuntimeError('Pre-spawn resource guard')
    child=subprocess.Popen([sys.executable,'-B','-X','pycache_prefix='+str(out/'_fresh_no_write_cache'),
        str(ROOT/'scripts/trackocd_cvpr/pilot_fresh_A_optimizer.py'),'--worker',
        '--preregistration-commit',args.preregistration_commit,'--git-proxy',args.git_proxy]
        + (['--recovery-attempt',args.recovery_attempt] if getattr(args,'recovery_attempt',None) else []),cwd=ROOT,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    try:
        while child.poll() is None:
            status_path=Path(f'/proc/{child.pid}/status')
            try:
                status=status_path.read_text(); rollup=Path(f'/proc/{child.pid}/smaps_rollup').read_text()
                peak=max(peak,process_memory(status,rollup)['guarded_process_RSS_bytes'])
                fields=dict(l.split(':',1) for l in status.splitlines() if ':' in l)
                threads=max(threads,int(fields['Threads'].strip()))
            except (FileNotFoundError,ProcessLookupError,RuntimeError):
                if child.poll() is not None: break
                try: terminal=kernel_terminal_status(status_path.read_text())
                except (FileNotFoundError,ProcessLookupError): terminal=False
                if terminal: child.wait(timeout=1); break
                if time.monotonic()-started>2:
                    reason='owned_memory_counter_unavailable'; child.kill(); break
                time.sleep(.2); continue
            reason=resources_ok(cfg,safety,ROOT,started,peak)
            if output_bytes(out)>cfg['output_bytes_ceiling']: reason='owned_output_size'
            try: driver_helpers=max(driver_helpers,inspect_owned_threads(child.pid))
            except ValueError: reason='owned_worker_threads'
            except (FileNotFoundError,ProcessLookupError):
                if child.poll() is not None: break
                reason='owned_thread_inventory_unavailable'
            if reason:
                if child.poll() is None: child.kill()  # ONLY our newly created Popen child
                break
            time.sleep(.2)
        code=child.wait(timeout=10)
    except BaseException:
        if child.poll() is None: child.kill()
        child.wait(timeout=10); raise
    if code==0 and reason is None and not (out/'result.json').is_file(): reason='owned_result_missing'
    atomic_json(receipt,{'owned_worker_exit_code':code,'guard_stop_reason':reason,
        'successful_owned_execution':code==0 and reason is None,'full_result_present':(out/'result.json').is_file(),
        'only_new_owned_Popen_worker_managed':True,'peak_sampled_owned_worker_RSS_bytes':peak,
        'peak_sampled_worker_threads':threads,'wall_seconds_including_imports_and_network':time.monotonic()-started,
        'peak_identified_native_driver_helpers':driver_helpers,'CPU_compute_workers':1,
        'training_GPU_hours':0,'execution_device':'CPU'})
    return 0 if code==0 and reason is None else 1


def worker(args,cfg,safety):
    out=ROOT/cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix!=str(out/'_fresh_no_write_cache'):
        raise ValueError('Fresh no-write CPU worker required')
    proof=verify_registration(ROOT,safety['branch'],args.preregistration_commit,args.git_proxy,allow_public_api=True)
    for p in SOURCES:
        if (ROOT/p).is_symlink() or subprocess.check_output(['git','show',args.preregistration_commit+':'+p],cwd=ROOT)!=(ROOT/p).read_bytes():
            raise ValueError('Exact registered pilot/source/authority/cache receipt required')
    validate_execution(cfg,json.loads((ROOT/cfg['approval']).read_text()))
    if cfg.get('CPU_visibility_recovery'): verify_original(ROOT)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('Owned CPU worker must hide GPUs before import')
    import torch
    from src.trackocd_cvpr.fresh_A_training import validate_protocol,fit_baseline,verify_matched_pilot,checkpoint_state_equal
    from src.trackocd_cvpr.fresh_A_baseline import FAMILIES,fresh_model
    from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks
    from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
    protocol=json.loads((ROOT/PROTOCOL).read_text()); validate_protocol(protocol)
    if out.exists(): raise ValueError('No pilot output overwrite')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    def barrier(event,values):
        if event=='socket.connect': raise PermissionError('No network after registration')
        if event=='open' and isinstance(values[0],(str,bytes)):
            p=Path(os.fsdecode(values[0])).resolve()
            if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')):
                raise PermissionError('Cache-only fit, no original GT or pixels')
    sys.addaudithook(barrier)
    out.mkdir(parents=True); started=time.monotonic(); peak=0
    prior=sum(p.stat().st_size for p in (ROOT/'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak=max(peak,process_memory()['guarded_process_RSS_bytes'])
        reason=resources_ok(cfg,safety,ROOT,started,peak); size=output_bytes(out)
        if reason or size>cfg['output_bytes_ceiling'] or prior+size>safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned baseline guard:'+str(reason or 'output_storage'))
        if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1 or torch.cuda.is_initialized():
            raise RuntimeError('Exactly one intra/inter-op CPU compute worker, no CUDA tensors')
    atomic_json(out/'launch.json',{'preregistration_commit':args.preregistration_commit,
        'live_remote_registration_proof':proof,'source_sha256':{p:sha256_file(ROOT/p) for p in SOURCES},
        'scope':'Fresh five-family fit5 pilot only, no formal warmstart/dev/scientific acceptance'})
    try:
        guard(); data_cfg=json.loads((ROOT/cfg['data_config']).read_text())
        tracks,label_proof=load_completed_fit_tracks(ROOT,data_cfg,guard); rows=[]; endpoints=[]
        for family in FAMILIES:
            def progress(row,curve):
                atomic_json(out/'progress.json',{'family':family,**row})
                atomic_json(out/(family+'_loss_curve.json'),curve)
            def save(model,optimizer,step,metadata):
                path=out/(family+'_private_step10_checkpoint.pt'); tmp=path.with_suffix('.pt.partial')
                with tmp.open('xb') as f:
                    os.chmod(tmp,0o600)
                    torch.save({'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                        'step':step,**metadata},f); f.flush(); os.fsync(f.fileno())
                if path.exists(): raise ValueError('Preserve endpoint checkpoint')
                os.replace(tmp,path)
                guard()
                restored=torch.load(path,map_location='cpu',weights_only=True)
                verifier=fresh_model(family,1027); verifier.load_state_dict(restored['model_state'],strict=True)
                if (digest(verifier)!=digest(model) or restored['step']!=10 or restored['stage']!='pilot'
                        or restored['family']!=family or restored['seed']!=1027
                        or not checkpoint_state_equal(restored['optimizer_state'],optimizer.state_dict())):
                    raise ValueError('Actual private checkpoint reopen differs')
                endpoints.append({'family':family,'bytes':path.stat().st_size,'sha256':sha256_file(path),'reopen_exact':True})
            model,optimizer,row=fit_baseline(tracks,protocol,family,1027,'pilot',guard,progress,save)
            rows.append(row); del model,optimizer; guard()
        parity=verify_matched_pilot(rows)
        if rows[0]['scope']['physical_tracks']!=1401 or rows[0]['scope']['eligible_known_physical_tracks']!=1158 or rows[0]['scope']['masked_no_fit5_target_tracks']!=243:
            raise ValueError('Complete actual fit physical denominator')
        result={'status':'COMPLETED_FRESH_FIVE_FAMILY_FIT5_PILOT_NOT_FORMAL_DEV_OR_GOAL_ACCEPTANCE',
            'live_remote_registration_proof':proof,'label_and_physical_scope':label_proof,'trials':rows,
            'actual_parity':parity,'private_checkpoints':endpoints,'formal_warmstart':False,
            'dev_final_Val_Test_inputs_used':False,'resources':{'wall_seconds_excluding_network':time.monotonic()-started,
                'peak_guarded_RSS_bytes':peak,'training_GPU_hours':0,'CUDA_initialized':torch.cuda.is_initialized()}}
        guard()
        if output_bytes(out)+len(json.dumps(result,indent=2).encode())+4096>cfg['output_bytes_ceiling']:
            raise RuntimeError('Final result would exceed owned output ceiling')
        atomic_json(out/'result.json',result); print(json.dumps(result),flush=True)
    except Exception as e:
        atomic_json(out/'failure.json',{'exception_type':type(e).__name__,'error':str(e),'complete':False}); raise


def main():
    global SOURCES
    p=argparse.ArgumentParser(); p.add_argument('--preregistration-commit',required=True)
    p.add_argument('--git-proxy',default=''); p.add_argument('--worker',action='store_true')
    p.add_argument('--recovery-attempt',choices=('cpu_visibility_retry1',)); args=p.parse_args()
    cfg=json.loads((ROOT/CONFIG).read_text()); safety=json.loads((ROOT/'configs/trackocd_cvpr/autonomous_research.json').read_text())
    validate_execution(cfg,json.loads((ROOT/cfg['approval']).read_text()))
    cfg=recovery_configuration(ROOT,cfg,args.recovery_attempt)
    if args.recovery_attempt: SOURCES=tuple(dict.fromkeys((*SOURCES,*RECOVERY_SOURCES)))
    if args.worker: worker(args,cfg,safety); return 0
    return supervise(args,cfg,safety)


if __name__=='__main__': sys.exit(main())
