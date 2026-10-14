#!/usr/bin/env python3
"""One explicitly registered read-only complete policy observer qualification."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration
from src.trackocd_cvpr.arbitration_attribution_protocol import configuration, CONFIG, REPORT
from src.trackocd_cvpr.raw_membership_formal_cohort import verified_cohort, directory, TRIALS
from src.trackocd_cvpr.raw_membership_development_receipts import trial_sources
from src.trackocd_cvpr.raw_membership_baseline_reference import HASHES
from scripts.trackocd_cvpr import evaluate_raw_membership_development as prior

SCRIPT = 'scripts/trackocd_cvpr/pilot_arbitration_attribution.py'
LAUNCHER = 'scripts/trackocd_cvpr/launch_owned_arbitration_attribution.py'
runtime = prior.runtime
SOURCES = tuple(dict.fromkeys((*prior.SOURCES,*(p for c,s in TRIALS for p in trial_sources(c,s)),*HASHES,
    CONFIG,REPORT,SCRIPT,LAUNCHER,'src/trackocd_cvpr/arbitration_attribution.py',
    'src/trackocd_cvpr/arbitration_attribution_protocol.py','src/trackocd_cvpr/raw_membership_development_receipts.py',
    'src/trackocd_cvpr/raw_membership_baseline_reference.py','src/trackocd_cvpr/raw_membership_control_comparison.py')))


def source_gate(args,cfg,safety):
    if cfg != configuration(ROOT):raise ValueError('Exact one frozen observer pilot before private input')
    proof = verify_registration(ROOT,safety['branch'],args.preregistration_commit,args.git_proxy,allow_public_api=True)
    for p in SOURCES:
        if (ROOT/p).is_symlink() or subprocess.check_output(['git','show',args.preregistration_commit+':'+p],cwd=ROOT) != (ROOT/p).read_bytes():
            raise ValueError('All qualified real ALL4/reference and new observer paths registered before private reads')
    a = json.loads((ROOT/cfg['approval']).read_text())
    if (a['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN' or a['video_scope'] != 407
            or a['allowed_training_label_roles'] != ['fit_known','policy_train_pseudo']
            or a['new_nonKnown_semantic_GT_or_Novel_vocabulary_authorized']
            or a['unlabeled_objects_background_or_inferred_Novel_labels']
            or a['old_51_heldout_Val_Test_DEVplus_Q1_public_new_sealed_and_other_videos_allowed']):
        raise ValueError('Actual human restricted fit/policy407 source authority')
    return proof


def busy():
    if prior.busy():return True
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) == os.getpid():continue
        try:words = p.read_bytes().split(b'\0'); state = (p.parent/'stat').read_text().rsplit(')',1)[1].split()[0]
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if os.fsencode(str(ROOT/SCRIPT)) in words and state not in ('Z','X','x'):return True
    return False


def supervise(args,cfg,safety):
    out = ROOT/cfg['output_directory']; terminal = out.with_name(out.name+'_supervisor_attempt_1.json')
    if out.exists() or terminal.exists():raise ValueError('Preserve every prior pilot namespace, no unchanged retry')
    if busy():raise RuntimeError('Existing owned research process live; observe, never duplicate')
    source_gate(args,cfg,safety); start = time.monotonic(); peak = threads = helpers = 0; reason = None
    mem = {k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}
    if (runtime.resources_ok(cfg,safety,ROOT,start,0)
            or mem['MemAvailable'] < safety['system_ram_reserve_fraction']*mem['MemTotal']+cfg['rss_bytes_ceiling']):
        raise RuntimeError('Original reserves plus planned RSS before own spawn')
    child = subprocess.Popen([sys.executable,'-B','-X','pycache_prefix='+str(out/'_fresh_no_write_cache'),
        str(ROOT/SCRIPT),'--worker','--preregistration-commit',args.preregistration_commit,'--git-proxy',args.git_proxy],cwd=ROOT,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
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
                if child.poll() is None:child.kill()  # ONLY this new owned Popen child.
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
    from src.trackocd_cvpr.arbitration_attribution import attribution_stream
    from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
    from src.trackocd_cvpr.R_development_evaluation import load_completed_fit_tracks, known_prototypes
    from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
    from src.trackocd_cvpr.raw_known_membership_control import NamedKnownLogit, parameter_sha256
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    allowed = {(ROOT/directory(c,s)/name).resolve() for c,s in TRIALS
        for name in ([f'private_step{k}_checkpoint.pt' for k in (20,50,100,200)] if c == 'calendar_matched' else ['private_full_policy_checkpoint.pt'])}
    def barrier(event,values):
        if event == 'socket.connect':raise PermissionError('No network after registration/authority gates')
        if event == 'open' and isinstance(values[0],(str,bytes)):
            p = Path(os.fsdecode(values[0])).resolve()
            if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (p.suffix in ('.pt','.pth','.partial') and p not in allowed):
                raise PermissionError('Qualified fit/policy caches and own ALL4 formal snapshots only; no pixels/old/external weights')
    sys.addaudithook(barrier)
    cohort = verified_cohort(ROOT,check_private_snapshots=True)
    if out.exists():raise ValueError('No prior pilot output overwrite')
    out.mkdir(parents=True); start = time.monotonic(); peak = 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT/'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak,runtime.process_memory()['guarded_process_RSS_bytes'])
        size = runtime.output_bytes(out); reason = runtime.resources_ok(cfg,safety,ROOT,start,peak)
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes+size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned attribution pilot guard:'+str(reason or 'output_storage'))
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1 or torch.cuda.is_initialized():
            raise RuntimeError('One CPU compute worker, no CUDA')
    atomic_json(out/'launch.json',{'preregistration_commit':args.preregistration_commit,'live_remote_registration_proof':proof,
        'source_sha256':{p:sha256_file(ROOT/p) for p in SOURCES},'scope':'All4 real full-dev required; all5 frozen heads/full policy clock; read-only observer+reference+ID/future probes, no fitting/intervention write'})
    try:
        guard(); videos,projection,scope = load_completed_policy_videos(ROOT,cfg,guard)
        fit,fit_scope = load_completed_fit_tracks(ROOT,cfg,guard)
        raw,_ = known_prototypes(RawPreservingEvidence().eval().requires_grad_(False),fit)
        known = {i:v for i,v in enumerate(raw)}
        targets = {(v,str(route)):label for (v,route),label in projection.targets.items()}
        models = [('initial_Raw',NamedKnownLogit().eval().requires_grad_(False))]
        for control,seed in TRIALS:
            row = next(r for r in cohort['trials'] if (r['control'],r['calendar_seed']) == (control,seed))
            p = ROOT/directory(control,seed)/('private_step200_checkpoint.pt' if control == 'calendar_matched' else 'private_full_policy_checkpoint.pt')
            state = torch.load(p,map_location='cpu',weights_only=True)
            if state['control'] != control or state['step'] != row['fixed_endpoint']['step']:raise ValueError('Exact frozen formal endpoint')
            h = NamedKnownLogit(); h.load_state_dict(state['model_state'],strict=True); h.eval().requires_grad_(False); del state
            if parameter_sha256(h) != row['model_parameter_sha256']:raise ValueError('Actual frozen formal parameters')
            models.append((f'calendar{seed}' if control == 'calendar_matched' else 'ONE_fullpolicy',h))
        if [name for name,_ in models] != cfg['models']:raise ValueError('All5 fixed models, no selected subset')
        results = {}
        for name,head in models:
            before = parameter_sha256(head)
            measured,private = attribution_stream(videos,targets,known,cfg,cfg['order_seed'],guard,head=head)
            normal,reference = attribution_stream(videos,targets,known,cfg,cfg['order_seed'],guard,head=head,observe=False)
            if private != reference or measured['full_clock'] != normal['full_clock']:raise ValueError('Observer changed actual event/score/commit/state trace')
            del reference
            renamed,renamed_private = attribution_stream(videos,targets,known,cfg,cfg['order_seed'],guard,head=head,rename_and_reverse=True)
            if (renamed != measured or [r.events for _,r in renamed_private[0]] != [r.events for _,r in private[0]]
                    or renamed_private[2] != private[2]):raise ValueError('Actual full policy ID/reverse inputs change numerical witness/score/outcome')
            del renamed_private
            ordinal = int(np.random.default_rng(cfg['order_seed']).permutation(len(videos))[0]); v,arrays = videos[ordinal]
            split = int(arrays['frame_index'].min()); future = arrays['frame_index'] > split
            if not future.any():raise ValueError('Nonvacuous future perturbation required')
            changed = {k:a.copy() for k,a in arrays.items()}; changed['visual'][future] = 0; changed['visual'][future,0] = 1
            mutations = int(np.any(changed['visual'][future] != arrays['visual'][future],axis=1).sum())
            if mutations == 0:raise ValueError('Actual future numeric observations must change')
            altered = list(videos); altered[ordinal] = (v,changed)
            _,altered_private = attribution_stream(altered,targets,known,cfg,cfg['order_seed'],guard,head=head)
            left = [r.events for video,r in private[0] if video == v and r.frame_index <= split]
            right = [r.events for video,r in altered_private[0] if video == v and r.frame_index <= split]
            if left != right or not any(left):raise ValueError('Nonempty earlier sealed events changed under future perturbation')
            if parameter_sha256(head) != before:raise ValueError('Frozen observer model changed')
            results[name] = {'measured':measured,'formal_parameter_sha256':before,'normal_event_commit_state_parity':True,
                'actual_full_ID_reverse_outcome_event_state_equal':True,'actual_future_prefix_equal':True,'future_observations_mutated':mutations}
            atomic_json(out/'progress.json',{'completed_model':name,'complete_result_present':False,'optimizer_steps':0})
            del private,altered_private
        guard(); result = {'status':'MEASURED_READONLY_POLICY_ATTRIBUTION_PILOT_NOT_UNSEEN_DEV_INTERVENTION_WIN_OR_GOAL',
            'live_remote_registration_proof':proof,'policy_scope':scope,'offline_fit_prior_scope':fit_scope,'models':results,
            'all13_private_snapshot_fingerprints_checked_before_policy':True,'optimizer_steps':0,
            'shadow_or_GT_used_for_policy_memory_write':False,'dev_final_Val_Test_pixels_or_new_nonKnown_semantics_read':False,
            'policy_resubstitution_is_unseen_or_joint_Goal_acceptance':False,
            'resources':{'peak_guarded_RSS_bytes':peak,'wall_seconds_excluding_network':time.monotonic()-start,
                         'training_GPU_hours':0,'CUDA_initialized':torch.cuda.is_initialized()}}
        atomic_json(out/'result.json',result); print(json.dumps({'status':result['status'],'completed_models':len(results)}),flush=True)
    except Exception as error:
        atomic_json(out/'failure.json',{'exception_type':type(error).__name__,'error':str(error),'complete':False}); raise


def parser():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit',required=True); p.add_argument('--git-proxy',default='')
    p.add_argument('--worker',action='store_true'); return p


def main():
    args = parser().parse_args(); cfg = configuration(ROOT); safety = json.loads((ROOT/'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker:worker(args,cfg,safety); return 0
    return supervise(args,cfg,safety)


if __name__ == '__main__':sys.exit(main())
