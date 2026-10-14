#!/usr/bin/env python3
"""One frozen head/all3 full Train-dev orders and independent shadow checks."""
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
from src.trackocd_cvpr.arbitration_development_protocol import configuration,CONFIG,PILOT_HASHES,PILOT_PROOF
from src.trackocd_cvpr.raw_membership_formal_cohort import verified_cohort,TRIALS,directory
from scripts.trackocd_cvpr import pilot_arbitration_attribution as pilot

SCRIPT = 'scripts/trackocd_cvpr/evaluate_arbitration_development.py'
LAUNCHER = 'scripts/trackocd_cvpr/launch_owned_arbitration_development.py'
runtime = pilot.runtime
SOURCES = tuple(dict.fromkeys((*pilot.SOURCES,*PILOT_HASHES,PILOT_PROOF,CONFIG,SCRIPT,LAUNCHER,
    'src/trackocd_cvpr/arbitration_attribution_qualification.py','src/trackocd_cvpr/arbitration_development_adapter.py',
    'src/trackocd_cvpr/arbitration_development_protocol.py')))


def source_gate(args,cfg,safety):
    if cfg != configuration(ROOT,args.model):raise ValueError('Exact one frozen-head complete development protocol')
    proof = verify_registration(ROOT,safety['branch'],args.preregistration_commit,args.git_proxy,allow_public_api=True)
    for p in SOURCES:
        if (ROOT/p).is_symlink() or subprocess.check_output(['git','show',args.preregistration_commit+':'+p],cwd=ROOT) != (ROOT/p).read_bytes():
            raise ValueError('All qualified numerical/source/input/private role boundaries registered before data')
    a = json.loads((ROOT/cfg['approval']).read_text())
    if (a['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN' or a['video_scope'] != 407
            or a['allowed_training_label_roles'] != ['fit_known','policy_train_pseudo']
            or not a['development_for_Train_development_only'] or a['independent_final_for_model_selection']
            or a['new_nonKnown_semantic_GT_or_Novel_vocabulary_authorized'] or a['unlabeled_objects_background_or_inferred_Novel_labels']
            or a['old_51_heldout_Val_Test_DEVplus_Q1_public_new_sealed_and_other_videos_allowed']):
        raise ValueError('Actual human407Train-development-only authority, no new semantic labels')
    return proof


def busy():
    if pilot.busy():return True
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) == os.getpid():continue
        try:words = p.read_bytes().split(b'\0'); state = (p.parent/'stat').read_text().rsplit(')',1)[1].split()[0]
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if os.fsencode(str(ROOT/SCRIPT)) in words and state not in ('Z','X','x'):return True
    return False


def supervise(args,cfg,safety):
    out = ROOT/cfg['output_directory']; terminal = out.with_name(out.name+'_supervisor_attempt_1.json')
    if out.exists() or terminal.exists():raise ValueError('Preserve all previous attempts, no unchanged retry')
    if busy():raise RuntimeError('Existing owned researcher live; observe, no parallel launch')
    source_gate(args,cfg,safety); start = time.monotonic(); peak = threads = helpers = 0; reason = None
    mem = {k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}
    if (runtime.resources_ok(cfg,safety,ROOT,start,0)
            or mem['MemAvailable'] < safety['system_ram_reserve_fraction']*mem['MemTotal']+cfg['rss_bytes_ceiling']):
        raise RuntimeError('Original reserves plus planned RSS before own spawn')
    child = subprocess.Popen([sys.executable,'-B','-X','pycache_prefix='+str(out/'_fresh_no_write_cache'),
        str(ROOT/SCRIPT),'--worker','--model',args.model,'--preregistration-commit',args.preregistration_commit,'--git-proxy',args.git_proxy],
        cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    try:
        while child.poll() is None:
            p = Path(f'/proc/{child.pid}/status')
            try:
                status = p.read_text(); peak = max(peak,runtime.process_memory(status,Path(f'/proc/{child.pid}/smaps_rollup').read_text())['guarded_process_RSS_bytes'])
                fields = dict(l.split(':',1) for l in status.splitlines() if ':' in l); threads = max(threads,int(fields['Threads']))
            except (FileNotFoundError,ProcessLookupError,RuntimeError):
                if child.poll() is not None:break
                try:settled = runtime.kernel_terminal_status(p.read_text())
                except (FileNotFoundError,ProcessLookupError):settled = False
                if settled:child.wait(timeout=1); break
                try:child.wait(timeout=.1); break
                except subprocess.TimeoutExpired:pass
                if time.monotonic()-start > 2:reason = 'owned_memory_counter_unavailable'; child.kill(); break
                time.sleep(.2); continue
            reason = runtime.resources_ok(cfg,safety,ROOT,start,peak)
            if runtime.output_bytes(out) > cfg['output_bytes_ceiling']:reason = 'owned_output_size'
            try:helpers = max(helpers,runtime.inspect_owned_threads(child.pid))
            except ValueError:reason = 'owned_worker_threads'
            except (FileNotFoundError,ProcessLookupError):
                if child.poll() is not None:break
                reason = 'owned_thread_inventory_unavailable'
            if reason:
                if child.poll() is None:child.kill()  # ONLY this fresh owned Popen child.
                break
            time.sleep(.2)
        code = child.wait(timeout=10)
    except BaseException:
        if child.poll() is None:child.kill()
        child.wait(timeout=10); raise
    complete = (out/'result.json').is_file()
    atomic_json(terminal,{'owned_worker_exit_code':code,'guard_stop_reason':reason,
        'successful_owned_execution':code == 0 and reason is None and complete,'complete_result_present':complete,
        'peak_sampled_owned_worker_RSS_bytes':peak,'peak_sampled_worker_threads':threads,
        'peak_identified_native_driver_helpers':helpers,'CPU_compute_workers':1,'training_GPU_hours':0,
        'only_new_owned_Popen_worker_managed':True,'wall_seconds_including_worker_imports_network':time.monotonic()-start})
    return 0 if code == 0 and reason is None and complete else 1


def worker(args,cfg,safety):
    out = ROOT/cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(out/'_fresh_no_write_cache') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Fresh hidden-GPU CPU observer worker')
    proof = source_gate(args,cfg,safety)
    import numpy as np
    import torch
    from src.trackocd_cvpr.arbitration_development_adapter import development_stream
    from src.trackocd_cvpr.C_development_rollout import dev_videos_and_external_labels
    from src.trackocd_cvpr.R_development_evaluation import load_completed_fit_tracks,known_prototypes
    from src.trackocd_cvpr.fit_loss_supervision import project_fit_loss_targets
    from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
    from src.trackocd_cvpr.raw_known_membership_control import NamedKnownLogit,parameter_sha256
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    allowed = {(ROOT/directory(c,s)/name).resolve() for c,s in TRIALS
        for name in ([f'private_step{k}_checkpoint.pt' for k in (20,50,100,200)] if c == 'calendar_matched' else ['private_full_policy_checkpoint.pt'])}
    def barrier(event,values):
        if event == 'socket.connect':raise PermissionError('No network after source/authority registration')
        if event == 'open' and isinstance(values[0],(str,bytes)):
            p = Path(os.fsdecode(values[0])).resolve()
            if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (p.suffix in ('.pt','.pth','.partial') and p not in allowed):
                raise PermissionError('Qualified caches and own ALL4 fixed formal snapshots only; no original pixels/annotations/oldweights')
    sys.addaudithook(barrier)
    cohort = verified_cohort(ROOT,check_private_snapshots=True)
    if out.exists():raise ValueError('No prior output overwrite')
    out.mkdir(parents=True); start = time.monotonic(); peak = 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT/'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak,runtime.process_memory()['guarded_process_RSS_bytes'])
        size = runtime.output_bytes(out); reason = runtime.resources_ok(cfg,safety,ROOT,start,peak)
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes+size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned complete development attribution guard:'+str(reason or 'output_storage'))
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1 or torch.cuda.is_initialized():
            raise RuntimeError('One CPU compute worker, no CUDA')
    atomic_json(out/'launch.json',{'preregistration_commit':args.preregistration_commit,'live_remote_registration_proof':proof,
        'source_sha256':{p:sha256_file(ROOT/p) for p in SOURCES},'model':args.model,
        'scope':'One all5-window frozen model/full72dev/3orders+each normal reference+fullID-future probes; shadow scores never written'})
    try:
        guard(); head = NamedKnownLogit()
        if args.model != 'initial_Raw':
            control,seed = ('full_policy',None) if args.model == 'ONE_fullpolicy' else ('calendar_matched',int(args.model[-4:]))
            row = next(r for r in cohort['trials'] if (r['control'],r['calendar_seed']) == (control,seed))
            p = ROOT/directory(control,seed)/('private_step200_checkpoint.pt' if control == 'calendar_matched' else 'private_full_policy_checkpoint.pt')
            state = torch.load(p,map_location='cpu',weights_only=True)
            if state['control'] != control or state['step'] != row['fixed_endpoint']['step']:raise ValueError('Exact actual fixed formal endpoint')
            head.load_state_dict(state['model_state'],strict=True); del state
            if parameter_sha256(head) != row['model_parameter_sha256']:raise ValueError('Actual unchanged formal head')
        head.eval().requires_grad_(False); before = parameter_sha256(head)
        videos,labels,scope = dev_videos_and_external_labels(ROOT,cfg,guard)
        fit,fit_scope = load_completed_fit_tracks(ROOT,cfg,guard)
        bindings,_,_ = project_fit_loss_targets(ROOT/cfg['private_assignment'],ROOT/cfg['private_Known_support'])
        raw,_ = known_prototypes(RawPreservingEvidence().eval().requires_grad_(False),fit); known = {i:v for i,v in enumerate(raw)}
        def run(data,order,observe=True,renamed=False):
            return development_stream(data,labels,known,bindings,cfg,order,guard,head=head,observe=observe,rename_and_reverse=renamed)
        results = {}; first = None
        for order in cfg['order_seeds']:
            measured,private = run(videos,order)
            normal,reference = run(videos,order,False)
            if private != reference or measured['full_clock'] != normal['full_clock']:raise ValueError('Observer changed normal full order events/commits/state')
            del reference
            results[str(order)] = {'measured':measured,'actual_full_normal_event_commit_state_equal':True}
            if first is None:first = private
            atomic_json(out/'progress.json',{'completed_order_with_normal_reference':order,'model':args.model,'complete_result_present':False,'optimizer_steps':0})
            if private is not first:del private
        renamed,renamed_private = run(videos,cfg['order_seeds'][0],renamed=True)
        if (renamed != results[str(cfg['order_seeds'][0])]['measured']
                or [r.events for _,r in renamed_private[0]] != [r.events for _,r in first[0]] or renamed_private[2] != first[2]):
            raise ValueError('Full ID/reverse changes observer numeric events/state/outcomes')
        del renamed_private
        ordinal = int(np.random.default_rng(cfg['order_seeds'][0]).permutation(len(videos))[0]); v,arrays = videos[ordinal]
        split = int(arrays['frame_index'].min()); future = arrays['frame_index'] > split
        if not future.any():raise ValueError('Nonvacuous complete future perturbation required')
        changed = {k:a.copy() for k,a in arrays.items()}; changed['visual'][future] = 0; changed['visual'][future,0] = 1
        mutations = int(np.any(changed['visual'][future] != arrays['visual'][future],axis=1).sum())
        if mutations == 0:raise ValueError('Future probe must change actual observations')
        altered = list(videos); altered[ordinal] = (v,changed); _,alternate = run(altered,cfg['order_seeds'][0])
        left = [r.events for video,r in first[0] if video == v and r.frame_index <= split]
        right = [r.events for video,r in alternate[0] if video == v and r.frame_index <= split]
        if left != right or not any(left):raise ValueError('Nonempty sealed earlier events changed under future mutation')
        if parameter_sha256(head) != before:raise ValueError('Frozen head changed')
        guard(); result = {'status':'MEASURED_COMPLETE_TRAIN_DEV_ARBITRATION_OBSERVER_NOT_INTERVENED_STREAM_JOINT_PRED_FINAL_OR_GOAL',
            'live_remote_registration_proof':proof,'model':args.model,'evaluated_parameter_sha256':before,'development_scope':scope,
            'offline_fit_prior_scope':fit_scope,'orders':results,'all13_snapshot_fingerprints_before_dev':True,
            'full_ID_reverse_events_outcomes_state_equal':True,'actual_future_probe':{'future_observations_mutated':mutations,'nonempty_prior_events_equal':True},
            'external_bucket_policy_pseudo_is_declared_dev_pseudo_not_training':True,
            'optimizer_steps':0,'shadow_or_GT_used_for_policy_memory_write':False,'dev_head_seed_order_threshold_selected':False,
            'final_Val_Test_original_pixels_or_new_nonKnown_semantics_read':False,'one_decision_shadows_are_full_intervention_gain_or_Goal_acceptance':False,
            'resources':{'peak_guarded_RSS_bytes':peak,'wall_seconds_excluding_network':time.monotonic()-start,'training_GPU_hours':0,'CUDA_initialized':torch.cuda.is_initialized()}}
        atomic_json(out/'result.json',result); print(json.dumps({'status':result['status'],'model':args.model}),flush=True)
    except Exception as error:
        atomic_json(out/'failure.json',{'exception_type':type(error).__name__,'error':str(error),'complete':False}); raise


def parser():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit',required=True); p.add_argument('--git-proxy',default='')
    p.add_argument('--model',choices=('initial_Raw','calendar1027','calendar1028','calendar1029','ONE_fullpolicy'),required=True)
    p.add_argument('--worker',action='store_true'); return p


def main():
    args = parser().parse_args(); cfg = configuration(ROOT,args.model); safety = json.loads((ROOT/'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker:worker(args,cfg,safety); return 0
    return supervise(args,cfg,safety)


if __name__ == '__main__':sys.exit(main())
