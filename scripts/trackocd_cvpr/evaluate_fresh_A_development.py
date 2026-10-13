#!/usr/bin/env python3
"""All15 native baseline full dev; no optimizer/selector/partial cohort."""
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
from src.trackocd_cvpr.fresh_A_formal_fit import cohort_sources,verified_cohort,PREFIX,active_research_workers
from scripts.trackocd_cvpr.fit_fresh_A_fixed200 import SOURCES as FIT_SOURCES
from scripts.trackocd_cvpr.evaluate_R_development import SOURCES as R_SOURCES
from scripts.trackocd_cvpr.pilot_fresh_A_optimizer import output_bytes
from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok,kernel_terminal_status
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory
from src.trackocd_cvpr.fresh_A_CPU_recovery import inspect_owned_threads

CONFIG='configs/trackocd_cvpr/fresh_A_development_all15_v1.json'
SCRIPT='scripts/trackocd_cvpr/evaluate_fresh_A_development.py'
SOURCES=tuple(dict.fromkeys((*FIT_SOURCES,*R_SOURCES,CONFIG,SCRIPT,
    'configs/trackocd_cvpr/R_development_first_pair_v1.json',
    'src/trackocd_cvpr/fresh_A_development_encoding.py','src/trackocd_cvpr/fresh_A_development_metrics.py',
    'src/trackocd_cvpr/fresh_A_development_probes.py')))


def configuration(root):
    c=json.loads((root/CONFIG).read_text())
    expected={'status':'REGISTERED_READONLY_FRESH_A_DEV_ALL15_FIXED_FITS_REQUIRED',
        'data_config':'configs/trackocd_cvpr/R_development_first_pair_v1.json',
        'families':['A1_ADAPTER','A1_GEOMETRY_1','A1_GEOMETRY_5','A2_EVIDENCE','A2_CAPACITY_CONTROL'],
        'training_seeds':[1027,1028,1029],'checkpoint_step':200,'prefixes':[1,2,4,8,16,'all_available'],
        'physical_tracks':472,'observations':8753,'metadata_frames':2562,'Known_queries':302,
        'declared_pseudo_queries':119,'outside_eligible_Known':2,'semantic_truth_unavailable':49,
        'optimizer_steps':0,'select_model_or_checkpoint':False,'execution_device':'CPU','CPU_compute_workers':1,
        'rss_bytes_ceiling':3221225472,'wall_seconds_ceiling':3000,'output_bytes_ceiling':8388608,
        'output_directory':'outputs/trackocd_cvpr/cvpr2/fresh_A_development_all15_v1',
        'all15_actual_normal_endpoints_before_private_dev_input':True,
        'final_Val_Test_new_nonKnown_GT_vocabulary_or_original_pixels_allowed':False,'full_joint_Pred_or_Goal_accepted':False}
    if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen complete native15 development protocol, not subset/selector')
    return {**json.loads((root/c['data_config']).read_text()),**c}


def source_gate(args,cfg,safety):
    # Complete public cohort before even network work; no actual data on missing15.
    cohort=verified_cohort(ROOT,check_private_endpoints=False)
    proof=verify_registration(ROOT,safety['branch'],args.preregistration_commit,args.git_proxy,allow_public_api=True)
    paths=(*SOURCES,*cohort_sources())
    for p in paths:
        if (ROOT/p).is_symlink() or subprocess.check_output(['git','show',args.preregistration_commit+':'+p],cwd=ROOT)!=(ROOT/p).read_bytes():
            raise ValueError('All45 public fits and exact source/cache/authority registration')
    if json.loads((ROOT/cfg['approval']).read_text())['status']!='USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN':
        raise ValueError('Actual scoped development authority')
    return proof,paths,cohort


def supervise(args,cfg,safety):
    out=ROOT/cfg['output_directory'];terminal=out.with_name(out.name+'_supervisor_attempt_1.json')
    if out.exists() or terminal.exists():raise ValueError('Preserve any previous dev attempt')
    if active_research_workers(ROOT):raise RuntimeError('Owned fit/M worker active; no parallel development')
    source_gate(args,cfg,safety);start=time.monotonic();peak=0;threads=0;reason=None
    if resources_ok(cfg,safety,ROOT,start,0):raise RuntimeError('Pre-spawn resource guard')
    child=subprocess.Popen([sys.executable,'-B','-X','pycache_prefix='+str(out/'_fresh_no_write_cache'),
        str(ROOT/SCRIPT),'--worker','--preregistration-commit',args.preregistration_commit,'--git-proxy',args.git_proxy],cwd=ROOT,
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
                try:settled=kernel_terminal_status(status_path.read_text())
                except (FileNotFoundError,ProcessLookupError):settled=False
                if settled:child.wait(timeout=1);break
                if time.monotonic()-start>2:reason='owned_memory_counter_unavailable';child.kill();break
                time.sleep(.2);continue
            reason=resources_ok(cfg,safety,ROOT,start,peak)
            if output_bytes(out)>cfg['output_bytes_ceiling']:reason='owned_output_size'
            try:inspect_owned_threads(child.pid)
            except ValueError:reason='owned_worker_threads'
            except (FileNotFoundError,ProcessLookupError):
                if child.poll() is not None:break
                reason='owned_thread_inventory_unavailable'
            if reason:
                if child.poll() is None:child.kill()  # ONLY our fresh Popen child
                break
            time.sleep(.2)
        code=child.wait(timeout=10)
    except BaseException:
        if child.poll() is None:child.kill()
        child.wait(timeout=10);raise
    if code==0 and reason is None and not (out/'result.json').is_file():reason='owned_result_missing'
    atomic_json(terminal,{'owned_worker_exit_code':code,'guard_stop_reason':reason,
        'successful_owned_execution':code==0 and reason is None,'complete_result_present':(out/'result.json').is_file(),
        'only_new_owned_Popen_worker_managed':True,'peak_sampled_owned_worker_RSS_bytes':peak,
        'peak_sampled_worker_threads':threads,'CPU_compute_workers':1,'training_GPU_hours':0,
        'wall_seconds_including_worker_imports_network':time.monotonic()-start})
    return 0 if code==0 and reason is None else 1


def worker(args,cfg,safety):
    out=ROOT/cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix!=str(out/'_fresh_no_write_cache') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Owned hidden-GPU no-write CPU worker')
    proof,paths,public=source_gate(args,cfg,safety)
    # Actual private endpoint hashes only after complete public/fresh source gate.
    cohort=verified_cohort(ROOT,check_private_endpoints=True)
    import torch
    from src.trackocd_cvpr.fresh_A_baseline import fresh_model
    from src.trackocd_cvpr.fresh_A_development_encoding import encode_native_prefixes,offline_fit5_native_prototypes
    from src.trackocd_cvpr.fresh_A_development_metrics import evaluate_prefix
    from src.trackocd_cvpr.fresh_A_development_probes import numeric_probes
    from src.trackocd_cvpr.R_development_evaluation import load_completed_dev_tracks,load_completed_fit_tracks,known_prototypes,embed_complete_role
    from src.trackocd_cvpr.fit_loss_supervision import project_fit_loss_targets
    from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
    from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(1027)
    allowed={(ROOT/f"{PREFIX}_{r['family']}_s{r['seed']}/private_step200_checkpoint.pt").resolve() for r in cohort['trials']}
    def barrier(event,values):
        if event=='socket.connect':raise PermissionError('No network after registration')
        if event=='open' and isinstance(values[0],(str,bytes)):
            p=Path(os.fsdecode(values[0])).resolve()
            if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (p.suffix in ('.pt','.pth') and p not in allowed):
                raise PermissionError('Only current qualified cache and all15 fixed own endpoints')
    sys.addaudithook(barrier)
    if out.exists():raise ValueError('No evaluation overwrite')
    out.mkdir(parents=True);start=time.monotonic();peak=0
    prior=sum(p.stat().st_size for p in (ROOT/'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak=max(peak,process_memory()['guarded_process_RSS_bytes']);size=output_bytes(out);reason=resources_ok(cfg,safety,ROOT,start,peak)
        if reason or size>cfg['output_bytes_ceiling'] or prior+size>safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned full15dev guard:'+str(reason or 'output_storage'))
        if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1 or torch.cuda.is_initialized():raise RuntimeError('CPU-only fixed endpoint evaluation')
    atomic_json(out/'launch.json',{'live_remote_registration_proof':proof,'source_sha256':{p:sha256_file(ROOT/p) for p in paths},
        'scope':'All15 fixed200 native baselines/full frozen dev, no optimizer or selector'})
    try:
        guard();tracks,labels,label_proof=load_completed_dev_tracks(ROOT,cfg,guard)
        fit,fit_proof=load_completed_fit_tracks(ROOT,cfg,guard)
        bindings,_,_=project_fit_loss_targets(ROOT/cfg['private_assignment'],ROOT/cfg['private_Known_support'])
        raw_model=RawPreservingEvidence().eval().requires_grad_(False);raw_prototypes,_=known_prototypes(raw_model,fit)
        raw_values={str(cap):embed_complete_role(raw_model,tracks,cap,guard) for cap in cfg['prefixes']}
        table={};raw_reference={};probes={}
        for row in cohort['trials']:
            family,seed=row['family'],row['seed'];directory=ROOT/f'{PREFIX}_{family}_s{seed}'
            checkpoint=directory/'private_step200_checkpoint.pt'
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
            model=fresh_model(family,seed);model.load_state_dict(payload['model_state'],strict=True);model.eval().requires_grad_(False)
            receipt=json.loads((directory/'result.json').read_text());before=digest(model)
            if (payload['step']!=200 or payload['stage']!='Fixed200' or payload['family']!=family or payload['seed']!=seed
                    or before!=receipt['trial']['model_parameter_sha256_after']):raise ValueError('Actual fixed200 native endpoint, not substituted weights')
            native_prototypes,prototype_proof=offline_fit5_native_prototypes(model,fit,guard);cases={}
            for cap in cfg['prefixes']:
                adapted,actual=encode_native_prefixes(model,tracks,cap,guard);raw,_,raw_actual=raw_values[str(cap)]
                if actual!=raw_actual:raise ValueError('Same complete observed prefix footprint')
                current=evaluate_prefix(raw,adapted,tracks,labels,bindings,raw_prototypes,native_prototypes,actual)
                if str(cap) in raw_reference and raw_reference[str(cap)]!=current['Raw']:raise ValueError('Frozen zero-fit Raw reference changed across15 baselines')
                raw_reference[str(cap)]=current['Raw'];cases[str(cap)]=current
                atomic_json(out/'progress.json',{'completed_family':family,'seed':seed,'prefix':str(cap),'physical_tracks':472,'no_optimizer':True})
            table.setdefault(family,{})[str(seed)]={'cases':cases,'prototype_scope':prototype_proof}
            probes.setdefault(family,{})[str(seed)]=numeric_probes(model,tracks,guard)
            if digest(model)!=before:raise ValueError('Read-only native endpoint changed')
            del model,payload;guard()
        result={'status':'MEASURED_ALL15_FRESH_GT_BASELINE_DEV_NOT_JOINT_PRED_FINAL_OR_GOAL_ACCEPTANCE',
            'live_remote_registration_proof':proof,'cohort':public,'development_scope':label_proof,'offline_fit_scope':fit_proof,
            'trials':table,'Raw_reference':raw_reference,'numeric_probes':probes,'optimizer_steps':0,'model_or_checkpoint_selected':False,
            'final_Val_Test_original_pixels_new_nonKnown_GT_or_vocabulary_used':False,'full_joint_Pred_or_Goal_accepted':False,
            'resources':{'peak_guarded_RSS_bytes':peak,'wall_seconds_excluding_network':time.monotonic()-start,'training_GPU_hours':0,'CUDA_initialized':torch.cuda.is_initialized()}}
        guard()
        if output_bytes(out)+len(json.dumps(result,indent=2).encode())+4096>cfg['output_bytes_ceiling']:raise RuntimeError('Full result output ceiling')
        atomic_json(out/'result.json',result);print(json.dumps({'status':result['status']}),flush=True)
    except Exception as e:
        atomic_json(out/'failure.json',{'exception_type':type(e).__name__,'error':str(e),'complete':False});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit',required=True);p.add_argument('--git-proxy',default='');p.add_argument('--worker',action='store_true');args=p.parse_args()
    cfg=configuration(ROOT);safety=json.loads((ROOT/'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker:worker(args,cfg,safety);return 0
    return supervise(args,cfg,safety)


if __name__=='__main__':sys.exit(main())
