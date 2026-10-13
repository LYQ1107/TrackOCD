#!/usr/bin/env python3
"""One gated fixed endpoint/full3orders plus complete ID/future probes; no fit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration
from src.trackocd_cvpr.spread_C_development_protocol import configuration, CONFIG
from src.trackocd_cvpr.spread_C_complete_fit_cohort import verified_cohort, cohort_sources
from src.trackocd_cvpr.spread_C_formal_protocol import PREFIX as FIT_PREFIX
from scripts.trackocd_cvpr.fit_spread_known_fixed200 import SOURCES as FIT_SOURCES, busy as fit_busy
from scripts.trackocd_cvpr.evaluate_C_development import BASE_SOURCES
from scripts.trackocd_cvpr import pilot_spread_known_rejection as runtime

SCRIPT = 'scripts/trackocd_cvpr/evaluate_spread_C_development.py'
LAUNCHER = 'scripts/trackocd_cvpr/launch_owned_spread_C_development.py'
SOURCES = tuple(dict.fromkeys((*FIT_SOURCES, *BASE_SOURCES, *cohort_sources(), CONFIG, SCRIPT, LAUNCHER,
    'src/trackocd_cvpr/spread_C_development_protocol.py', 'src/trackocd_cvpr/spread_C_complete_fit_cohort.py',
    'outputs/trackocd_cvpr/cvpr3/paired_spread_Known_rejection_all6_fit_receipt_v1.json')))


def source_gate(args, cfg, safety):
    actual = configuration(ROOT, args.arm, args.seed)
    if cfg != actual:
        raise ValueError('Unchanged explicit full-development configuration')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for path in SOURCES:
        if ((ROOT/path).is_symlink() or subprocess.check_output(
                ['git', 'show', args.preregistration_commit+':'+path], cwd=ROOT) != (ROOT/path).read_bytes()):
            raise ValueError('All source/full actual fit cohort preregistered before private input')
    if json.loads((ROOT/cfg['approval']).read_text())['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN':
        raise ValueError('Real scoped training/development authority')
    cohort = verified_cohort(ROOT)
    public = json.loads((ROOT/'outputs/trackocd_cvpr/cvpr3/paired_spread_Known_rejection_all6_fit_receipt_v1.json').read_text())
    if {k:public[k] for k in cohort} != cohort:
        raise ValueError('Public all6 receipt exactly binds complete original fits')
    return proof


def busy():
    if fit_busy():
        return True
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) == os.getpid():
            continue
        try:
            words = p.read_bytes().split(b'\0'); state = (p.parent/'stat').read_text().rsplit(')', 1)[1].split()[0]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if os.fsencode(str(ROOT/SCRIPT)) in words and state not in ('Z', 'X', 'x'):
            return True
    return False


def supervise(args, cfg, safety):
    out = ROOT/cfg['output_directory']; terminal = out.with_name(out.name+'_supervisor_attempt_1.json')
    if out.exists() or terminal.exists():
        raise ValueError('Never overwrite/retry a completed or incomplete evaluator namespace')
    if busy():
        raise RuntimeError('Observe live owned evaluator/fit/M, never duplicate')
    source_gate(args, cfg, safety)
    start = time.monotonic(); peak = threads = helpers = 0; reason = None
    mem = {k:int(v.split()[0])*1024 for k,v in
           (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    if (runtime.resources_ok(cfg, safety, ROOT, start, 0)
            or mem['MemAvailable'] < safety['system_ram_reserve_fraction']*mem['MemTotal']+cfg['rss_bytes_ceiling']):
        raise RuntimeError('Pre-spawn reserve plus full planned RSS')
    child = subprocess.Popen([sys.executable, '-B', '-X', 'pycache_prefix='+str(out/'_fresh_no_write_cache'),
        str(ROOT/SCRIPT), '--worker', '--arm', args.arm, '--seed', str(args.seed),
        '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy], cwd=ROOT,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
    try:
        while child.poll() is None:
            p = Path(f'/proc/{child.pid}/status')
            try:
                status = p.read_text()
                peak = max(peak, runtime.process_memory(status,
                    Path(f'/proc/{child.pid}/smaps_rollup').read_text())['guarded_process_RSS_bytes'])
                fields = dict(l.split(':', 1) for l in status.splitlines() if ':' in l)
                threads = max(threads, int(fields['Threads']))
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                if child.poll() is not None:
                    break
                try:
                    settled = runtime.kernel_terminal_status(p.read_text())
                except (FileNotFoundError, ProcessLookupError):
                    settled = False
                if settled:
                    child.wait(timeout=1); break
                try:
                    child.wait(timeout=.1); break
                except subprocess.TimeoutExpired:
                    pass
                if time.monotonic()-start > 2:
                    reason = 'owned_memory_counter_unavailable'; child.kill(); break
                time.sleep(.2); continue
            reason = runtime.resources_ok(cfg, safety, ROOT, start, peak)
            if runtime.output_bytes(out) > cfg['output_bytes_ceiling']:
                reason = 'owned_output_size'
            try:
                helpers = max(helpers, runtime.inspect_owned_threads(child.pid))
            except ValueError:
                reason = 'owned_worker_threads'
            except (FileNotFoundError, ProcessLookupError):
                if child.poll() is not None:
                    break
                reason = 'owned_thread_inventory_unavailable'
            if reason:
                if child.poll() is None:
                    child.kill()  # ONLY this fresh owned Popen child, never other work.
                break
            time.sleep(.2)
        code = child.wait(timeout=10)
    except BaseException:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10); raise
    complete = (out/'result.json').is_file()
    atomic_json(terminal, {'owned_worker_exit_code':code, 'guard_stop_reason':reason,
        'successful_owned_execution':code == 0 and reason is None and complete, 'complete_result_present':complete,
        'peak_sampled_owned_worker_RSS_bytes':peak, 'peak_sampled_worker_threads':threads,
        'peak_identified_native_driver_helpers':helpers, 'CPU_compute_workers':1, 'training_GPU_hours':0,
        'only_new_owned_Popen_worker_managed':True, 'wall_seconds_including_worker_imports_network':time.monotonic()-start})
    return 0 if code == 0 and reason is None and complete else 1


def worker(args, cfg, safety):
    out = ROOT/cfg['output_directory']
    if (not sys.dont_write_bytecode or sys.pycache_prefix != str(out/'_fresh_no_write_cache')
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise ValueError('Fresh hidden-GPU no-write evaluation worker')
    proof = source_gate(args, cfg, safety)
    import numpy as np
    import torch
    from src.trackocd_cvpr.C_development_rollout import dev_videos_and_external_labels, readonly_C_stream
    from src.trackocd_cvpr.R_development_evaluation import load_completed_fit_tracks, known_prototypes
    from src.trackocd_cvpr.fit_loss_supervision import project_fit_loss_targets
    from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
    from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
    from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    allowed = {(ROOT/f'{FIT_PREFIX}_s{s}'/f'private_{a}_step200.pt').resolve()
               for s in cfg['training_seeds'] for a in cfg['trained_arms']}
    def barrier(event, values):
        if event == 'socket.connect':
            raise PermissionError('No network after registration')
        if event == 'open' and isinstance(values[0], (str, bytes)):
            p = Path(os.fsdecode(values[0])).resolve()
            if (p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal'))
                    or (p.suffix in ('.pt', '.pth', '.partial') and p not in allowed)):
                raise PermissionError('Qualified cache and own ALL6 Fixed200 endpoints only, no old/pilot/external weights')
    sys.addaudithook(barrier)
    # Source/live/authority gates completed; all endpoint fingerprints before dev inputs.
    cohort = verified_cohort(ROOT, check_private_endpoints=True)
    selected = next(r for r in cohort['trials'] if r['seed'] == args.seed and r['arm'] == args.arm)
    checkpoint = ROOT/f'{FIT_PREFIX}_s{args.seed}'/f'private_{args.arm}_step200.pt'
    if out.exists():
        raise ValueError('No prior output overwrite')
    out.mkdir(parents=True); start = time.monotonic(); peak = 0
    prior = sum(p.stat().st_size for p in (ROOT/'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, runtime.process_memory()['guarded_process_RSS_bytes'])
        size = runtime.output_bytes(out); reason = runtime.resources_ok(cfg, safety, ROOT, start, peak)
        if reason or size > cfg['output_bytes_ceiling'] or prior+size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned full-development guard:'+str(reason or 'output_storage'))
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1 or torch.cuda.is_initialized():
            raise RuntimeError('One CPU compute worker/no CUDA')
    atomic_json(out/'launch.json', {'preregistration_commit':args.preregistration_commit,
        'live_remote_registration_proof':proof, 'source_sha256':{p:sha256_file(ROOT/p) for p in SOURCES},
        'arm':args.arm, 'seed':args.seed, 'endpoint':selected['endpoint'],
        'scope':'ALL6 real fits required, one read-only endpoint/complete72dev videos/all3orders/ID+future probes'})
    try:
        guard(); state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['arm'] != args.arm or state['seed'] != args.seed or state['step'] != 200:
            raise ValueError('Exact fixed200 own arm/seed endpoint, no selected stage')
        model = CandidateScoreMLP(); model.load_state_dict(state['model_state'], strict=True)
        model.eval().requires_grad_(False)
        if digest(model) != selected['endpoint_parameter_sha256']:
            raise ValueError('Exact actually trained endpoint parameters')
        del state
        videos, labels, scope = dev_videos_and_external_labels(ROOT, cfg, guard)
        fit, fit_scope = load_completed_fit_tracks(ROOT, cfg, guard)
        bindings, _, _ = project_fit_loss_targets(ROOT/cfg['private_assignment'], ROOT/cfg['private_Known_support'])
        raw, _ = known_prototypes(RawPreservingEvidence().eval().requires_grad_(False), fit)
        known = {i:row for i,row in enumerate(raw)}; known_bindings = {slot:category for category,slot in bindings.items()}
        table = {}; first = None
        def run(inputs, order, renamed=False):
            return readonly_C_stream(inputs, labels, known, known_bindings, cfg,
                cfg['learned_numeric_interface'], order, guard, model=model, rename_and_reverse=renamed)
        for order in cfg['order_seeds']:
            measured, private = run(videos, order); table[str(order)] = measured
            if first is None:
                first = private
            atomic_json(out/'progress.json', {'completed_order':order, 'arm':args.arm, 'training_seed':args.seed,
                'full_observations':8753, 'full_frames':2562, 'optimizer_steps':0, 'complete_result_present':False})
            if private is not first:
                del private
        renamed, renamed_private = run(videos, cfg['order_seeds'][0], True)
        if (renamed != table[str(cfg['order_seeds'][0])]
                or [r.events for _,r in renamed_private] != [r.events for _,r in first]):
            raise ValueError('Actual full-stream bijective IDs/reversed delivery changed frozen outcome or trace')
        del renamed_private
        ordinal = int(np.random.default_rng(cfg['order_seeds'][0]).permutation(len(videos))[0])
        v, arrays = videos[ordinal]; split = int(arrays['frame_index'].min())
        future = arrays['frame_index'] > split
        if not future.any():
            raise ValueError('Nonvacuous actual future probe required, no unavailable substitution')
        changed = {k:a.copy() for k,a in arrays.items()}
        changed['visual'][future] = 0; changed['visual'][future,0] = 1
        mutations = int(np.any(changed['visual'][future] != arrays['visual'][future], axis=1).sum())
        if mutations == 0:
            raise ValueError('Future mutation must change actual observations')
        altered = list(videos); altered[ordinal] = (v, changed)
        _, altered_private = run(altered, cfg['order_seeds'][0])
        left = [r.events for video,r in first if video == v and r.frame_index <= split]
        right = [r.events for video,r in altered_private if video == v and r.frame_index <= split]
        if left != right or not any(left):
            raise ValueError('Actual nonempty earlier sealed prefix changed under future perturbation')
        guard()
        result = {'status':'MEASURED_FULL_CAUSAL_SPREAD_PAIRED_C_DEV_NOT_CALIBRATED_JOINT_PRED_FINAL_OR_GOAL',
            'live_remote_registration_proof':proof, 'arm':args.arm, 'training_seed':args.seed,
            'all6_fixed200_endpoint_fingerprints_checked':True, 'evaluated_endpoint':selected['endpoint'],
            'evaluated_model_parameter_sha256':digest(model), 'development_scope':scope,
            'offline_fit_prior_scope':fit_scope, 'orders':table, 'actual_full_ID_rename_reverse_stream_equal':True,
            'actual_future_probe':{'future_observations_mutated':mutations, 'verified_same_prior_events':True},
            'optimizer_steps':0, 'model_or_checkpoint_selection':False,
            'final_Val_Test_or_new_nonKnown_semantic_values_read':False,
            'calibrated_Raw_or_coverage_latency_match_claimed':False,
            'full_semantic_truth_Hscore_CT_joint_Pred_or_Goal_accepted':False,
            'resources':{'peak_guarded_RSS_bytes':peak, 'wall_seconds_excluding_network':time.monotonic()-start,
                         'training_GPU_hours':0, 'CUDA_initialized':torch.cuda.is_initialized()}}
        atomic_json(out/'result.json', result)
        print(json.dumps({'status':result['status'], 'arm':args.arm, 'seed':args.seed}), flush=True)
    except Exception as error:
        atomic_json(out/'failure.json', {'exception_type':type(error).__name__, 'error':str(error), 'complete':False})
        raise


def parser():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit', required=True)
    p.add_argument('--git-proxy', default=''); p.add_argument('--worker', action='store_true')
    p.add_argument('--arm', choices=('baseline', 'Known_set_rejection'), required=True)
    p.add_argument('--seed', type=int, choices=(1027, 1028, 1029), required=True)
    return p


def main():
    args = parser().parse_args(); cfg = configuration(ROOT, args.arm, args.seed)
    safety = json.loads((ROOT/'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker:
        worker(args, cfg, safety); return 0
    return supervise(args, cfg, safety)


if __name__ == '__main__':
    sys.exit(main())
