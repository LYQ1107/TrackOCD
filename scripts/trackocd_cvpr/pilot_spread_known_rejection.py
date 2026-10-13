#!/usr/bin/env python3
"""One gated fresh paired50 pilot/full policy clock, no formal/dev auto-start."""
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
from src.trackocd_cvpr.paired_known_rejection_protocol import source_window, CONFIG
from src.trackocd_cvpr.C_development_complete_matrix import cohort_sources, EXTRA_SOURCES
from scripts.trackocd_cvpr.pilot_candidate_optimizer import SOURCES as BASE
from scripts.trackocd_cvpr.evaluate_C_development import BASE_SOURCES
from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok, kernel_terminal_status
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory
from src.trackocd_cvpr.fresh_A_CPU_recovery import inspect_owned_threads
from src.trackocd_cvpr.fresh_A_formal_fit import active_research_workers
from scripts.trackocd_cvpr.pilot_fresh_A_optimizer import output_bytes

SCRIPT = 'scripts/trackocd_cvpr/pilot_spread_known_rejection.py'
OUT = 'outputs/trackocd_cvpr/cvpr3/paired_spread_Known_rejection_pilot50_v1'
DATA = 'configs/trackocd_cvpr/candidate_optimizer_pilot_v1.json'
SOURCES = tuple(dict.fromkeys((*BASE,*BASE_SOURCES,*cohort_sources(),*EXTRA_SOURCES,CONFIG,SCRIPT,
    'src/trackocd_cvpr/paired_known_rejection_protocol.py','src/trackocd_cvpr/paired_known_rejection_training.py',
    'src/trackocd_cvpr/complementary_known_loss.py','src/trackocd_cvpr/candidate_scientific_training.py',
    'src/trackocd_cvpr/C_development_complete_matrix.py','src/trackocd_cvpr/C_terminal_counter_qualification.py',
    'src/trackocd_cvpr/fresh_A_training.py','src/trackocd_cvpr/fresh_A_CPU_recovery.py',
    'src/trackocd_cvpr/fresh_A_formal_fit.py','scripts/trackocd_cvpr/pilot_fresh_A_optimizer.py',
    'outputs/trackocd_cvpr/cvpr3/C_development_all9_matrix_v1/result.json',
    'outputs/trackocd_cvpr/cvpr3/C_historical_loss_plan_audit_v1/result.json',
    'outputs/trackocd_cvpr/cvpr3/C_historical_loss_plan_audit_v1_supervisor_attempt_1.json')))


SOURCES=tuple(dict.fromkeys((*SOURCES,
    'scripts/trackocd_cvpr/pilot_paired_known_rejection.py',
    'configs/trackocd_cvpr/C_spread_known_exposure_pilot_v1.json',
    'src/trackocd_cvpr/spread_known_calendar.py','src/trackocd_cvpr/paired_spread_known_rejection_training.py',
    'outputs/trackocd_cvpr/cvpr3/paired_Known_set_rejection_pilot50_v1/result.json',
    'outputs/trackocd_cvpr/cvpr3/Known_anchor_opportunity_audit_v1_counter_completion_qualification_v1.json')))


def configuration(root):
    return {**json.loads((root/DATA).read_text()),**source_window(root),'output_directory':OUT,
        'approval':'configs/trackocd_cvpr/formal_research_scope_approval_v1.json'}


def busy():
    if active_research_workers(ROOT): return True
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name)==os.getpid(): continue
        try:
            words=p.read_bytes().split(b'\0'); state=(p.parent/'stat').read_text().rsplit(')',1)[1].split()[0]
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        paired=(ROOT/SCRIPT,ROOT/'scripts/trackocd_cvpr/pilot_paired_known_rejection.py')
        if any(os.fsencode(str(path)) in words for path in paired) and state not in ('Z','X','x'): return True
    return False


def source_gate(args,cfg,safety):
    source_window(ROOT)  # Public complete old matrices first; not a private read.
    spread=json.loads((ROOT/'configs/trackocd_cvpr/C_spread_known_exposure_pilot_v1.json').read_text())
    expected={'status':'REGISTERED_ONE_FIXED_EXPOSURE_CORRECTION_PILOT_NO_FORMAL_AUTO_START',
        'rule':'MIDPOINT_UNIFORM_KNOWN_ELIGIBLE_RANKS_NO_PSEUDO_OR_PREDICTION_HISTORY_SELECTOR',
        'seed':1027,'episodes':1,'optimizer_steps_each':50,'repair_coefficient':.25,
        'models_loss_Raw_data_order_prototypes_optimizer_or_history_rules_changed':False,
        'pseudo_or_dev_metric_based_frame_ranking_or_search':False,
        'coefficient_seed_steps_search_or_pilot_warmstart':False,'maximum_exposure_correction_pilot_attempts':1,
        'require_actual_nonzero_rejection_gradient_for_later_registration':True,'formal_auto_start_or_Goal_acceptance':False}
    if any(spread.get(k)!=v for k,v in expected.items()): raise ValueError('Fixed one exposure correction, no search/scope change')
    for path,key,expected_hash in ((
        'outputs/trackocd_cvpr/cvpr3/paired_Known_set_rejection_pilot50_v1/result.json','old_pilot_result_sha256',
        '9a67853e3070dec7d86f7b7b0b7638e9f68b48d374950bcd490692f6a58a8239'),(
        'outputs/trackocd_cvpr/cvpr3/Known_anchor_opportunity_audit_v1_counter_completion_qualification_v1.json',
        'opportunity_qualification_sha256','932d3dbd1d3d6b5cdc1f58aec0cb08e5fc6ba56a0939665237a94c7defa41db2')):
        if spread[key]!=expected_hash or (ROOT/path).is_symlink() or sha256_file(ROOT/path)!=expected_hash:
            raise ValueError('Actual original uninformative pilot and explicit qualified exposure evidence')

    proof=verify_registration(ROOT,safety['branch'],args.preregistration_commit,args.git_proxy,allow_public_api=True)
    for path in SOURCES:
        if (ROOT/path).is_symlink() or subprocess.check_output(['git','show',args.preregistration_commit+':'+path],cwd=ROOT)!=(ROOT/path).read_bytes():
            raise ValueError('Exact registered source/public prerequisites before private input')
    authority=json.loads((ROOT/cfg['approval']).read_text())
    if authority['status']!='USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN': raise ValueError('Real scoped training authority')
    qualified=ROOT/'outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1'
    if sha256_file(qualified/'result.json')!=cfg['policy_input_qualification_sha256']:
        raise ValueError('Exact full-policy input qualification')
    t=json.loads(qualified.with_name(qualified.name+'_supervisor_attempt_1.json').read_text())
    if not t['successful_owned_execution'] or t['owned_worker_exit_code']!=0 or t['guard_stop_reason'] is not None:
        raise ValueError('Actual normal input qualification')
    return proof


def supervise(args,cfg,safety):
    out=ROOT/OUT; terminal=out.with_name(out.name+'_supervisor_attempt_1.json')
    if out.exists() or terminal.exists(): raise ValueError('Preserve every prior paired attempt')
    if busy(): raise RuntimeError('Observe live owned research worker, no parallel paired pilot')
    source_gate(args,cfg,safety); start=time.monotonic(); peak=threads=helpers=0; reason=None
    if resources_ok(cfg,safety,ROOT,start,0): raise RuntimeError('Pre-spawn resource guard')
    child=subprocess.Popen([sys.executable,'-B','-X','pycache_prefix='+str(out/'_fresh_no_write_cache'),
        str(ROOT/SCRIPT),'--worker','--preregistration-commit',args.preregistration_commit,'--git-proxy',args.git_proxy],
        cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    try:
        while child.poll() is None:
            p=Path(f'/proc/{child.pid}/status')
            try:
                status=p.read_text(); peak=max(peak,process_memory(status,Path(f'/proc/{child.pid}/smaps_rollup').read_text())['guarded_process_RSS_bytes'])
                fields=dict(l.split(':',1) for l in status.splitlines() if ':' in l); threads=max(threads,int(fields['Threads']))
            except (FileNotFoundError,ProcessLookupError,RuntimeError):
                if child.poll() is not None: break
                try: settled=kernel_terminal_status(p.read_text())
                except (FileNotFoundError,ProcessLookupError): settled=False
                if settled: child.wait(timeout=1); break
                if time.monotonic()-start>2: reason='owned_memory_counter_unavailable'; child.kill(); break
                time.sleep(.2); continue
            reason=resources_ok(cfg,safety,ROOT,start,peak)
            if output_bytes(out)>cfg['output_bytes_ceiling']: reason='owned_output_size'
            try: helpers=max(helpers,inspect_owned_threads(child.pid))
            except ValueError: reason='owned_worker_threads'
            except (FileNotFoundError,ProcessLookupError):
                if child.poll() is not None: break
                reason='owned_thread_inventory_unavailable'
            if reason:
                if child.poll() is None: child.kill()  # ONLY this fresh Popen child.
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


def worker(args,cfg,safety):
    out=ROOT/OUT
    if not sys.dont_write_bytecode or sys.pycache_prefix!=str(out/'_fresh_no_write_cache') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Fresh hidden-GPU no-write CPU worker')
    proof=source_gate(args,cfg,safety)
    import torch
    from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
    from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks
    from src.trackocd_cvpr.R_development_evaluation import known_prototypes
    from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
    from src.trackocd_cvpr.paired_spread_known_rejection_training import paired_candidate_fit
    from src.trackocd_cvpr.spread_known_calendar import freeze_calendar
    from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
    from src.trackocd_cvpr.fresh_A_training import checkpoint_state_equal
    from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    def barrier(event,values):
        if event=='socket.connect': raise PermissionError('No network after registration')
        if event=='open' and isinstance(values[0],(str,bytes)):
            p=Path(os.fsdecode(values[0])).resolve()
            if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (p.suffix in ('.pt','.pth','.partial') and p.parent!=out):
                raise PermissionError('Qualified caches/own checkpoints only, no original pixels/GT or old weights')
    sys.addaudithook(barrier)
    if out.exists(): raise ValueError('No output overwrite')
    out.mkdir(parents=True); start=time.monotonic(); peak=0
    prior=sum(p.stat().st_size for p in (ROOT/'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak=max(peak,process_memory()['guarded_process_RSS_bytes']); size=output_bytes(out)
        reason=resources_ok(cfg,safety,ROOT,start,peak)
        if reason or size>cfg['output_bytes_ceiling'] or prior+size>safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned paired pilot guard:'+str(reason or 'output_storage'))
        if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1 or torch.cuda.is_initialized():
            raise RuntimeError('One CPU compute worker/no CUDA')
    atomic_json(out/'launch.json',{'preregistration_commit':args.preregistration_commit,
        'live_remote_registration_proof':proof,'source_sha256':{p:sha256_file(ROOT/p) for p in SOURCES},
        'scope':'One fixed uniform-rank spread Known plan/seed1027/fresh pair50/full policy; no formal/dev auto-start'})
    try:
        guard(); videos,projection,policy=load_completed_policy_videos(ROOT,cfg,guard)
        fit,fit_proof=load_completed_fit_tracks(ROOT,cfg,guard)
        raw,_=known_prototypes(RawPreservingEvidence().eval().requires_grad_(False),fit)
        known={i:v for i,v in enumerate(raw)}; saved=[]
        def save(arm,model,optimizer,step,episode,episode_step,rng):
            path=out/f'private_{arm}_step{step}.pt'; tmp=path.with_suffix('.partial')
            with tmp.open('xb') as f:
                os.chmod(tmp,0o600); torch.save({'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                    'arm':arm,'step':step,'episode':episode,'episode_step':episode_step,'numpy_rng_state':rng,
                    'torch_rng_state':torch.get_rng_state()},f); f.flush(); os.fsync(f.fileno())
            if path.exists(): raise ValueError('Preserve private checkpoint')
            os.replace(tmp,path); guard(); value=torch.load(path,map_location='cpu',weights_only=True)
            with torch.random.fork_rng(devices=[]): verifier=CandidateScoreMLP()
            verifier.load_state_dict(value['model_state'],strict=True)
            if (digest(verifier)!=digest(model) or value['arm']!=arm or value['step']!=50
                    or not checkpoint_state_equal(value['optimizer_state'],optimizer.state_dict())
                    or value['numpy_rng_state']!=rng or not torch.equal(value['torch_rng_state'],torch.get_rng_state())):
                raise ValueError('Exact actual model/AdamW/RNG checkpoint reopen')
            saved.append({'arm':arm,'step':step,'sha256':sha256_file(path),'bytes':path.stat().st_size,'reopen_exact':True})
        def progress(rows,curves): atomic_json(out/'progress.json',rows); atomic_json(out/'loss_curves.json',curves)
        frozen=freeze_calendar(videos,projection,1027,1,50,guard)
        atomic_json(out/'private_frozen_calendar.json',frozen)
        os.chmod(out/'private_frozen_calendar.json',0o600)
        models,optimizers,summary,private=paired_candidate_fit(videos,projection,known,1027,
            episodes=1,updates_per_episode=50,checkpoint_steps=[50],guard=guard,progress=progress,checkpoint=save,frozen_calendar=frozen)
        if (summary['optimizer_steps_each']!=50 or len(saved)!=2
                or any(e['videos']!=10 or e['metadata_frames']!=374 or e['physical_observations']!=1268 or e['empty_frames']!=1
                    for episodes in summary['episodes'].values() for e in episodes)):
            raise ValueError('Both full policy clocks/fixed50 normal optimizer paths')
        informative=any(r['rejection_parameter_gradient_norm']>0 for r in summary['loss_curves']['Known_set_rejection'])
        result={'status':'COMPLETED_SPREAD_PAIRED50_PILOT_NOT_FORMAL_DEV_CALIBRATION_OR_GOAL_ACCEPTANCE',
            'live_remote_registration_proof':proof,'policy_scope':policy,'offline_fit_scope':fit_proof,'summary':summary,
            'private_checkpoints':saved,'actual_nonzero_rejection_gradient':informative,
            'pilot_informative_for_potential_later_formal_registration':informative,
            'pilot_warmstart_formal_automatic_start_or_dev_selection':False,
            'raw_private_history_or_GT_exported':False,'dev_final_Val_Test_inputs_used':False,
            'resources':{'peak_guarded_RSS_bytes':peak,'wall_seconds_excluding_network':time.monotonic()-start,
                'training_GPU_hours':0,'CUDA_initialized':torch.cuda.is_initialized()}}
        guard(); atomic_json(out/'result.json',result)
        print(json.dumps({'status':result['status'],'informative':informative}),flush=True)
    except Exception as e:
        atomic_json(out/'failure.json',{'exception_type':type(e).__name__,'error':str(e),'complete':False}); raise


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preregistration-commit',required=True)
    p.add_argument('--git-proxy',default=''); p.add_argument('--worker',action='store_true'); args=p.parse_args()
    cfg=configuration(ROOT); safety=json.loads((ROOT/'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker: worker(args,cfg,safety); return 0
    return supervise(args,cfg,safety)


if __name__=='__main__': sys.exit(main())
