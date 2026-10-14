#!/usr/bin/env python3
"""Explicit one-head whole-policy origin-aware closed loops; never auto training."""
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
from src.trackocd_cvpr.continuity_policy_protocol import configuration,CONFIG,SOURCE_CONFIG,REPORT,EXPECTED
from src.trackocd_cvpr.arbitration_development_receipts import trial_sources
from src.trackocd_cvpr.raw_membership_formal_cohort import verified_cohort,TRIALS,directory
from scripts.trackocd_cvpr import evaluate_arbitration_development as prior

SCRIPT='scripts/trackocd_cvpr/pilot_continuity_reuse_policy.py'
LAUNCHER='scripts/trackocd_cvpr/launch_owned_continuity_policy.py'
runtime=prior.runtime
SOURCES=tuple(dict.fromkeys((*prior.SOURCES,*(p for m in EXPECTED['models'] for p in trial_sources(m)),
    CONFIG,SOURCE_CONFIG,REPORT,SCRIPT,LAUNCHER,'src/trackocd_cvpr/continuity_policy_protocol.py',
    'src/trackocd_cvpr/continuity_reuse_control.py','src/trackocd_cvpr/independent_support_view.py',
    'src/trackocd_cvpr/continuity_origin_audit.py','src/trackocd_cvpr/continuity_policy_rollout.py',
    'src/trackocd_cvpr/causal_category_audit.py','src/trackocd_cvpr/arbitration_development_receipts.py',
    'src/trackocd_cvpr/arbitration_development_comparison.py')))


def source_gate(args,cfg,safety):
    if cfg!=configuration(ROOT,args.model):raise ValueError('Exact complete one-head policy protocol before private input')
    proof=verify_registration(ROOT,safety['branch'],args.preregistration_commit,args.git_proxy,allow_public_api=True)
    for p in SOURCES:
        if (ROOT/p).is_symlink() or subprocess.check_output(['git','show',args.preregistration_commit+':'+p],cwd=ROOT)!=(ROOT/p).read_bytes():
            raise ValueError('All original qualified and new numerical/source/scope paths registered first')
    a=json.loads((ROOT/cfg['approval']).read_text())
    if (a['status']!='USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN' or a['video_scope']!=407
            or a['allowed_training_label_roles']!=['fit_known','policy_train_pseudo']
            or a['new_nonKnown_semantic_GT_or_Novel_vocabulary_authorized']
            or a['unlabeled_objects_background_or_inferred_Novel_labels']
            or a['old_51_heldout_Val_Test_DEVplus_Q1_public_new_sealed_and_other_videos_allowed']):
        raise ValueError('Actual human restricted407 fit/policy authority')
    return proof


def busy():
    if prior.busy():return True
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name)==os.getpid():continue
        try:words=p.read_bytes().split(b'\0');state=(p.parent/'stat').read_text().rsplit(')',1)[1].split()[0]
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if os.fsencode(str(ROOT/SCRIPT)) in words and state not in ('Z','X','x'):return True
    return False


def supervise(args,cfg,safety):
    out=ROOT/cfg['output_directory'];terminal=out.with_name(out.name+'_supervisor_attempt_1.json')
    if out.exists() or terminal.exists():raise ValueError('Preserve every attempt; no unchanged retry')
    if busy():raise RuntimeError('Owned researcher live, observe without duplicate')
    source_gate(args,cfg,safety);start=time.monotonic();peak=threads=helpers=0;reason=None
    mem={k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}
    if (runtime.resources_ok(cfg,safety,ROOT,start,0)
            or mem['MemAvailable']<safety['system_ram_reserve_fraction']*mem['MemTotal']+cfg['rss_bytes_ceiling']):
        raise RuntimeError('Original reserves plus planned RSS before own spawn')
    child=subprocess.Popen([sys.executable,'-B','-X','pycache_prefix='+str(out/'_fresh_no_write_cache'),
        str(ROOT/SCRIPT),'--worker','--model',args.model,'--preregistration-commit',args.preregistration_commit,'--git-proxy',args.git_proxy],
        cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    try:
        while child.poll() is None:
            p=Path(f'/proc/{child.pid}/status')
            try:
                status=p.read_text();peak=max(peak,runtime.process_memory(status,Path(f'/proc/{child.pid}/smaps_rollup').read_text())['guarded_process_RSS_bytes'])
                fields=dict(x.split(':',1) for x in status.splitlines() if ':' in x);threads=max(threads,int(fields['Threads']))
            except (FileNotFoundError,ProcessLookupError,RuntimeError):
                if child.poll() is not None:break
                try:settled=runtime.kernel_terminal_status(p.read_text())
                except (FileNotFoundError,ProcessLookupError):settled=False
                if settled:child.wait(timeout=1);break
                try:child.wait(timeout=.1);break
                except subprocess.TimeoutExpired:pass
                if time.monotonic()-start>2:reason='owned_memory_counter_unavailable';child.kill();break
                time.sleep(.2);continue
            reason=runtime.resources_ok(cfg,safety,ROOT,start,peak)
            if runtime.output_bytes(out)>cfg['output_bytes_ceiling']:reason='owned_output_size'
            try:helpers=max(helpers,runtime.inspect_owned_threads(child.pid))
            except ValueError:reason='owned_worker_threads'
            except (FileNotFoundError,ProcessLookupError):
                if child.poll() is not None:break
                reason='owned_thread_inventory_unavailable'
            if reason:
                if child.poll() is None:child.kill()  # ONLY this new Popen worker.
                break
            time.sleep(.2)
        code=child.wait(timeout=10)
    except BaseException:
        if child.poll() is None:child.kill()
        child.wait(timeout=10);raise
    complete=(out/'result.json').is_file()
    atomic_json(terminal,{'owned_worker_exit_code':code,'guard_stop_reason':reason,
        'successful_owned_execution':code==0 and reason is None and complete,'complete_result_present':complete,
        'peak_sampled_owned_worker_RSS_bytes':peak,'peak_sampled_worker_threads':threads,
        'peak_identified_native_driver_helpers':helpers,'CPU_compute_workers':1,'training_GPU_hours':0,
        'only_new_owned_Popen_worker_managed':True,'wall_seconds_including_worker_imports_network':time.monotonic()-start})
    return 0 if code==0 and reason is None and complete else 1


def worker(args,cfg,safety):
    out=ROOT/cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix!=str(out/'_fresh_no_write_cache') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Fresh hidden-GPU oneCPU policy worker')
    proof=source_gate(args,cfg,safety)
    import numpy as np
    import torch
    from src.trackocd_cvpr.continuity_policy_rollout import policy_stream,trace_numeric,trace_known_rebound
    from src.trackocd_cvpr.arbitration_attribution import attribution_stream
    from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
    from src.trackocd_cvpr.R_development_evaluation import load_completed_fit_tracks,known_prototypes
    from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
    from src.trackocd_cvpr.raw_known_membership_control import NamedKnownLogit,parameter_sha256
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    allowed={(ROOT/directory(c,s)/n).resolve() for c,s in TRIALS
        for n in ([f'private_step{k}_checkpoint.pt' for k in (20,50,100,200)] if c=='calendar_matched' else ['private_full_policy_checkpoint.pt'])}
    def barrier(event,values):
        if event=='socket.connect':raise PermissionError('No network after source/authority gates')
        if event=='open' and isinstance(values[0],(str,bytes)):
            p=Path(os.fsdecode(values[0])).resolve()
            if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (p.suffix in ('.pt','.pth','.partial') and p not in allowed):
                raise PermissionError('Qualified fit/policy caches and own formal endpoints only')
    sys.addaudithook(barrier);cohort=verified_cohort(ROOT,check_private_snapshots=True)
    if out.exists():raise ValueError('No earlier pilot overwrite')
    out.mkdir(parents=True);start=time.monotonic();peak=0
    prior_bytes=sum(p.stat().st_size for p in (ROOT/'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak=max(peak,runtime.process_memory()['guarded_process_RSS_bytes'])
        size=runtime.output_bytes(out);reason=runtime.resources_ok(cfg,safety,ROOT,start,peak)
        if reason or size>cfg['output_bytes_ceiling'] or prior_bytes+size>safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned continuity policy guard:'+str(reason or 'output_storage'))
        if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1 or torch.cuda.is_initialized():
            raise RuntimeError('One CPU/no CUDA required')
    atomic_json(out/'launch.json',{'model':args.model,'preregistration_commit':args.preregistration_commit,
        'live_remote_registration_proof':proof,'source_sha256':{p:sha256_file(ROOT/p) for p in SOURCES},
        'scope':'One frozen head/full4arms/full3orders+each original baseline+ID/future/label probes; actual own commits, origin-aware external audit only'})
    try:
        guard();head=NamedKnownLogit()
        if args.model!='initial_Raw':
            control,seed=('full_policy',None) if args.model=='ONE_fullpolicy' else ('calendar_matched',int(args.model[-4:]))
            row=next(r for r in cohort['trials'] if (r['control'],r['calendar_seed'])==(control,seed))
            path=ROOT/directory(control,seed)/('private_step200_checkpoint.pt' if control=='calendar_matched' else 'private_full_policy_checkpoint.pt')
            state=torch.load(path,map_location='cpu',weights_only=True)
            if state['control']!=control or state['step']!=row['fixed_endpoint']['step']:raise ValueError('Exact original formal endpoint')
            head.load_state_dict(state['model_state'],strict=True);del state
            if parameter_sha256(head)!=row['model_parameter_sha256']:raise ValueError('Original head unchanged')
        head.eval().requires_grad_(False);before=parameter_sha256(head)
        videos,projection,scope=load_completed_policy_videos(ROOT,cfg,guard)
        fit,fit_scope=load_completed_fit_tracks(ROOT,cfg,guard)
        raw,_=known_prototypes(RawPreservingEvidence().eval().requires_grad_(False),fit);known={i:v for i,v in enumerate(raw)}
        bindings={slot:category for category,slot in projection.fit_bindings.items()}
        pseudo={slot:category for category,slot in projection.pseudo_bindings.items()};labels={}
        for video,a in videos:
            for track in a['track_id']:
                kind,slot=projection.targets.get((video,int(track)),('no_retained_semantic_truth',None))
                labels[(video,str(int(track)))]=( ('Known',bindings[slot]) if kind=='Known'
                    else ('declared_pseudoNovel',pseudo[slot]) if kind=='policy_pseudo' else (kind,None))
        old_targets={(v,str(t)):label for (v,t),label in projection.targets.items()}
        def run(mode,order,*,data=None,truth=None,renamed=False,prior_known=None,prior_bindings=None):
            return policy_stream(videos if data is None else data,labels if truth is None else truth,
                known if prior_known is None else prior_known,bindings if prior_bindings is None else prior_bindings,
                cfg,mode,order,guard,head=head,rename_and_reverse=renamed)
        results={}
        for mode in cfg['arms']:
            orders={};first=None
            for order in cfg['order_seeds']:
                measured,private=run(mode,order)
                parity=None
                if mode=='full_raw':
                    normal,reference=attribution_stream(videos,old_targets,known,cfg,order,guard,head=head,observe=False)
                    if ([(x.video,x.result) for x in private[0]]!=reference[0] or private[1:]!=reference[1:]
                            or measured['full_clock']!=normal['full_clock'] or measured['allocated_tokens']!=normal['allocated_tokens']):
                        raise ValueError('FullRaw baseline changed original events/commits/state/allocator')
                    parity=True;del reference
                orders[str(order)]={'measured':measured,'actual_original_baseline_parity':parity}
                if first is None:first=private
                else:del private
                atomic_json(out/'progress.json',{'model':args.model,'completed_mode':mode,'completed_order':order,'complete_result_present':False})
            renamed,other=run(mode,cfg['order_seeds'][0],renamed=True)
            if (renamed!=orders[str(cfg['order_seeds'][0])]['measured'] or trace_numeric(first[0])!=trace_numeric(other[0]) or first[2]!=other[2]):
                raise ValueError('Full ID/reverse changes origins/events/numeric history/state/outcome')
            del other
            slots=sorted(known);mapping={k:slots[(i+1)%len(slots)] for i,k in enumerate(slots)}
            if len(slots)<2 or all(k==v for k,v in mapping.items()):raise ValueError('Nonvacuous actual Known binding shuffle')
            rebound,other=run(mode,cfg['order_seeds'][0],prior_known={mapping[k]:v for k,v in known.items()},
                prior_bindings={mapping[k]:v for k,v in bindings.items()})
            if (rebound!=orders[str(cfg['order_seeds'][0])]['measured']
                    or trace_numeric(first[0])!=trace_known_rebound(other[0],{v:k for k,v in mapping.items()}) or first[1:]!=other[1:]):
                raise ValueError('External Known binding permutation changed numeric decisions/history/outcomes')
            del other
            ordinal=int(np.random.default_rng(cfg['order_seeds'][0]).permutation(len(videos))[0]);v,a=videos[ordinal]
            split=int(a['frame_index'].min());future=a['frame_index']>split
            changed={k:x.copy() for k,x in a.items()};changed['visual'][future]=0;changed['visual'][future,0]=1
            mutations=int(np.any(changed['visual'][future]!=a['visual'][future],axis=1).sum())
            if not future.any() or not mutations:raise ValueError('Nonvacuous actual future mutation required')
            altered=list(videos);altered[ordinal]=(v,changed);_,other=run(mode,cfg['order_seeds'][0],data=altered)
            earlier=lambda trace:trace_numeric(tuple(x for x in trace if x.video==v and x.result.frame_index<=split))
            if not earlier(first[0]) or not any(x.result.events for x in first[0] if x.video==v and x.result.frame_index<=split) or earlier(first[0])!=earlier(other[0]):
                raise ValueError('Earlier actual event/origin/history/state changed from future mutation')
            del other
            changed_labels={};named=sorted(bindings.values());heldout=sorted(pseudo.values())
            if min(len(named),len(heldout))<2:raise ValueError('Actual nonvacuous external label swap')
            for key,(kind,category) in labels.items():
                pool=named if kind=='Known' else heldout if kind=='declared_pseudoNovel' else None
                changed_labels[key]=(kind,pool[(pool.index(category)+1)%len(pool)]) if pool else (kind,None)
            if changed_labels==labels:raise ValueError('External labels must actually change')
            _,other=run(mode,cfg['order_seeds'][0],truth=changed_labels)
            if trace_numeric(first[0])!=trace_numeric(other[0]) or first[1:]!=other[1:]:raise ValueError('External GT label swap changed actual writes')
            results[mode]={'orders':orders,'actual_full_ID_reverse_equal':True,'actual_full_Known_binding_shuffle_equal':True,'future_mutations':mutations,
                'nonempty_prior_event_origin_history_state_equal':True,'actual_external_label_swap_numeric_writes_equal':True}
            del first,other
        if parameter_sha256(head)!=before:raise ValueError('Frozen head changed')
        guard();result={'status':'MEASURED_COMPLETE_ONE_HEAD_FOUR_ARM_POLICY_CLOSED_LOOPS_NOT_UNSEEN_DEV_JOINT_PRED_OR_GOAL',
            'model':args.model,'live_remote_registration_proof':proof,'formal_parameter_sha256':before,
            'policy_scope':scope,'offline_fit_prior_scope':fit_scope,'arms':results,
            'all13_snapshot_fingerprints_before_policy':True,'optimizer_steps':0,'GT_or_future_truth_to_memory_repair':False,
            'dev_final_Val_Test_pixels_or_new_nonKnown_semantics_read':False,'policy_resubstitution_is_unseen_dev_or_Goal_gain':False,
            'resources':{'peak_guarded_RSS_bytes':peak,'wall_seconds_excluding_network':time.monotonic()-start,
                         'training_GPU_hours':0,'CUDA_initialized':torch.cuda.is_initialized()}}
        atomic_json(out/'result.json',result);print(json.dumps({'status':result['status'],'model':args.model}),flush=True)
    except Exception as error:
        atomic_json(out/'failure.json',{'exception_type':type(error).__name__,'error':str(error),'complete':False});raise


def parser():
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit',required=True);p.add_argument('--git-proxy',default='')
    p.add_argument('--model',required=True,choices=EXPECTED['models']);p.add_argument('--worker',action='store_true');return p


def main():
    args=parser().parse_args();cfg=configuration(ROOT,args.model);safety=json.loads((ROOT/'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker:worker(args,cfg,safety);return 0
    return supervise(args,cfg,safety)


if __name__=='__main__':sys.exit(main())
