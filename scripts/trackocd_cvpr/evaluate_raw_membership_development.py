#!/usr/bin/env python3
"""One explicit frozen formal head, complete dev diagnostics/causal probes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration
from src.trackocd_cvpr.raw_membership_development_protocol import configuration, CONFIG, FIT_RECEIPT
from src.trackocd_cvpr.raw_membership_formal_cohort import verified_cohort, cohort_sources, directory, TRIALS
from scripts.trackocd_cvpr import fit_raw_known_membership_formal as formal

SCRIPT = 'scripts/trackocd_cvpr/evaluate_raw_membership_development.py'
LAUNCHER = 'scripts/trackocd_cvpr/launch_owned_raw_membership_development.py'
runtime = formal.pilot.prior.runtime
SOURCES = tuple(dict.fromkeys((*formal.SOURCES, *cohort_sources(), CONFIG, FIT_RECEIPT, SCRIPT, LAUNCHER,
    'src/trackocd_cvpr/raw_membership_formal_cohort.py',
    'src/trackocd_cvpr/raw_membership_development.py',
    'src/trackocd_cvpr/raw_membership_development_protocol.py')))


def source_gate(args, cfg, safety):
    if cfg != configuration(ROOT, args.control, args.seed):
        raise ValueError('Exact explicit ALL4 frozen-head full-development protocol')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for p in SOURCES:
        if (ROOT/p).is_symlink() or subprocess.check_output(['git', 'show', args.preregistration_commit+':'+p], cwd=ROOT) != (ROOT/p).read_bytes():
            raise ValueError('All original formal/public-cohort and new dev sources registered BEFORE private input')
    a = json.loads((ROOT/cfg['approval']).read_text())
    if (a['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN' or a['video_scope'] != 407
            or a['allowed_training_label_roles'] != ['fit_known', 'policy_train_pseudo']
            or not a['development_for_Train_development_only'] or a['independent_final_for_model_selection']
            or a['new_nonKnown_semantic_GT_or_Novel_vocabulary_authorized']
            or a['unlabeled_objects_background_or_inferred_Novel_labels']
            or a['old_51_heldout_Val_Test_DEVplus_Q1_public_new_sealed_and_other_videos_allowed']):
        raise ValueError('Actual human407Train-development-only authority; no new semantic labels')
    return proof


def busy():
    if formal.busy(): return True
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) == os.getpid(): continue
        try:
            words = p.read_bytes().split(b'\0'); state = (p.parent/'stat').read_text().rsplit(')', 1)[1].split()[0]
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if os.fsencode(str(ROOT/SCRIPT)) in words and state not in ('Z', 'X', 'x'): return True
    return False


def explicit_args(args):
    return ['--control', args.control] + (['--seed', str(args.seed)] if args.seed is not None else [])


def supervise(args, cfg, safety):
    out = ROOT/cfg['output_directory']; terminal = out.with_name(out.name+'_supervisor_attempt_1.json')
    if out.exists() or terminal.exists(): raise ValueError('One attempt; preserve all completed/incomplete namespaces')
    if busy(): raise RuntimeError('Existing owned research worker live; observe, never duplicate')
    source_gate(args, cfg, safety)
    start = time.monotonic(); peak = threads = helpers = 0; reason = None
    mem = {k: int(v.split()[0])*1024 for k, v in (x.split(':', 1) for x in Path('/proc/meminfo').read_text().splitlines())}
    if (runtime.resources_ok(cfg, safety, ROOT, start, 0)
            or mem['MemAvailable'] < safety['system_ram_reserve_fraction']*mem['MemTotal']+cfg['rss_bytes_ceiling']):
        raise RuntimeError('Original reserve plus planned RSS before own spawn')
    child = subprocess.Popen([sys.executable, '-B', '-X', 'pycache_prefix='+str(out/'_fresh_no_write_cache'),
        str(ROOT/SCRIPT), '--worker', '--preregistration-commit', args.preregistration_commit,
        '--git-proxy', args.git_proxy, *explicit_args(args)], cwd=ROOT,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
    try:
        while child.poll() is None:
            p = Path(f'/proc/{child.pid}/status')
            try:
                status = p.read_text(); peak = max(peak, runtime.process_memory(status,
                    Path(f'/proc/{child.pid}/smaps_rollup').read_text())['guarded_process_RSS_bytes'])
                fields = dict(l.split(':', 1) for l in status.splitlines() if ':' in l); threads = max(threads, int(fields['Threads']))
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                if child.poll() is not None: break
                try: settled = runtime.kernel_terminal_status(p.read_text())
                except (FileNotFoundError, ProcessLookupError): settled = False
                if settled: child.wait(timeout=1); break
                try: child.wait(timeout=.1); break
                except subprocess.TimeoutExpired: pass
                if time.monotonic()-start > 2: reason = 'owned_memory_counter_unavailable'; child.kill(); break
                time.sleep(.2); continue
            reason = runtime.resources_ok(cfg, safety, ROOT, start, peak)
            if runtime.output_bytes(out) > cfg['output_bytes_ceiling']: reason = 'owned_output_size'
            try: helpers = max(helpers, runtime.inspect_owned_threads(child.pid))
            except ValueError: reason = 'owned_worker_threads'
            except (FileNotFoundError, ProcessLookupError):
                if child.poll() is not None: break
                reason = 'owned_thread_inventory_unavailable'
            if reason:
                if child.poll() is None: child.kill()  # ONLY this fresh owned Popen child.
                break
            time.sleep(.2)
        code = child.wait(timeout=10)
    except BaseException:
        if child.poll() is None: child.kill()
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
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(out/'_fresh_no_write_cache') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Fresh hidden-GPU no-write-cache CPU worker')
    proof = source_gate(args, cfg, safety)
    import numpy as np
    import torch
    from src.trackocd_cvpr.C_development_rollout import dev_videos_and_external_labels
    from src.trackocd_cvpr.R_development_evaluation import load_completed_fit_tracks, known_prototypes
    from src.trackocd_cvpr.fit_loss_supervision import project_fit_loss_targets
    from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
    from src.trackocd_cvpr.raw_known_membership_control import NamedKnownLogit, parameter_sha256
    from src.trackocd_cvpr.raw_membership_development import readonly_membership_stream, full_development_reliability
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    allowed = {(ROOT/directory(c, s)/name).resolve() for c, s in TRIALS
        for name in ([f'private_step{step}_checkpoint.pt' for step in (20, 50, 100, 200)]
                     if c == 'calendar_matched' else ['private_full_policy_checkpoint.pt'])}
    def barrier(event, values):
        if event == 'socket.connect': raise PermissionError('No network after source/authority registration')
        if event == 'open' and isinstance(values[0], (str, bytes)):
            p = Path(os.fsdecode(values[0])).resolve()
            if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (p.suffix in ('.pt', '.pth', '.partial') and p not in allowed):
                raise PermissionError('Qualified caches and own ALL4 formal snapshots only, no old/pilot/pixels/annotations')
    sys.addaudithook(barrier)
    cohort = verified_cohort(ROOT, check_private_snapshots=True)  # ALL13 actual bytes before dev input.
    selected = next(r for r in cohort['trials'] if (r['control'], r['calendar_seed']) == (args.control, args.seed))
    checkpoint = ROOT/directory(args.control, args.seed)/('private_step200_checkpoint.pt'
        if args.control == 'calendar_matched' else 'private_full_policy_checkpoint.pt')
    if out.exists(): raise ValueError('No output overwrite or unchanged retry')
    out.mkdir(parents=True); start = time.monotonic(); peak = 0
    prior = sum(p.stat().st_size for p in (ROOT/'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, runtime.process_memory()['guarded_process_RSS_bytes'])
        size = runtime.output_bytes(out); reason = runtime.resources_ok(cfg, safety, ROOT, start, peak)
        if reason or size > cfg['output_bytes_ceiling'] or prior+size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned full development guard:'+str(reason or 'output_storage'))
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1 or torch.cuda.is_initialized():
            raise RuntimeError('One CPU compute worker, no CUDA')
    atomic_json(out/'launch.json', {'preregistration_commit':args.preregistration_commit,
        'live_remote_registration_proof':proof, 'source_sha256':{p:sha256_file(ROOT/p) for p in SOURCES},
        'control':args.control, 'calendar_seed':args.seed, 'fixed_endpoint':selected['fixed_endpoint'],
        'scope':'ALL4 real fits/ALL13 bytes, one frozen endpoint/full72dev clocks/3orders/ID-reverse-future/conditional diagnostics'})
    try:
        guard(); state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['control'] != args.control or state['step'] != selected['fixed_endpoint']['step']:
            raise ValueError('Own actual fixed formal endpoint, not selected stage/pilot')
        head = NamedKnownLogit(); head.load_state_dict(state['model_state'], strict=True)
        head.eval().requires_grad_(False); del state
        if parameter_sha256(head) != selected['model_parameter_sha256']:
            raise ValueError('Actual unchanged formal endpoint parameters')
        initial = NamedKnownLogit().eval().requires_grad_(False)
        videos, labels, scope = dev_videos_and_external_labels(ROOT, cfg, guard)
        fit, fit_scope = load_completed_fit_tracks(ROOT, cfg, guard)
        bindings, _, _ = project_fit_loss_targets(ROOT/cfg['private_assignment'], ROOT/cfg['private_Known_support'])
        raw, _ = known_prototypes(RawPreservingEvidence().eval().requires_grad_(False), fit)
        known = {i: row for i, row in enumerate(raw)}; known_bindings = {slot: category for category, slot in bindings.items()}
        reliability = full_development_reliability(videos, labels, known, cfg, guard, head=head, initial=initial)
        table = {}; first = None
        def run(data, order, renamed=False):
            return readonly_membership_stream(data, labels, known, known_bindings, cfg, order,
                guard, head=head, rename_and_reverse=renamed)
        for order in cfg['order_seeds']:
            measured, private = run(videos, order); table[str(order)] = measured
            if first is None: first = private
            atomic_json(out/'progress.json', {'completed_order':order, 'control':args.control, 'calendar_seed':args.seed,
                'full_observations':8753, 'full_frames':2562, 'optimizer_steps':0, 'complete_result_present':False})
            if private is not first: del private
        renamed, renamed_private = run(videos, cfg['order_seeds'][0], True)
        if renamed != table[str(cfg['order_seeds'][0])] or [r.events for _, r in renamed_private] != [r.events for _, r in first]:
            raise ValueError('Actual full IDs/reversed delivery changed outcome or event trace')
        del renamed_private
        ordinal = int(np.random.default_rng(cfg['order_seeds'][0]).permutation(len(videos))[0])
        v, arrays = videos[ordinal]; split = int(arrays['frame_index'].min()); future = arrays['frame_index'] > split
        if not future.any(): raise ValueError('Nonvacuous complete future perturbation required')
        changed = {k: a.copy() for k, a in arrays.items()}; changed['visual'][future] = 0; changed['visual'][future, 0] = 1
        mutations = int(np.any(changed['visual'][future] != arrays['visual'][future], axis=1).sum())
        if mutations == 0: raise ValueError('Future probe must change actual observations')
        altered = list(videos); altered[ordinal] = (v, changed)
        _, alternate = run(altered, cfg['order_seeds'][0])
        left = [r.events for video, r in first if video == v and r.frame_index <= split]
        right = [r.events for video, r in alternate if video == v and r.frame_index <= split]
        if left != right or not any(left): raise ValueError('Nonempty sealed earlier events changed under future mutation')
        if parameter_sha256(head) != selected['model_parameter_sha256']: raise ValueError('Frozen model changed')
        guard()
        result = {'status':'MEASURED_FULL_RAW_MEMBERSHIP_DEV_NOT_SELECTED_JOINT_PRED_FINAL_OR_GOAL',
            'live_remote_registration_proof':proof, 'control':args.control, 'calendar_seed':args.seed,
            'all4_fits_and_all13_private_snapshot_fingerprints_checked':True,
            'evaluated_endpoint':selected['fixed_endpoint'], 'evaluated_model_parameter_sha256':parameter_sha256(head),
            'development_scope':scope, 'offline_fit_prior_scope':fit_scope, 'conditional_reliability':reliability,
            'orders':table, 'actual_full_ID_rename_reverse_stream_equal':True,
            'actual_future_probe':{'future_observations_mutated':mutations, 'verified_same_prior_events':True},
            'optimizer_steps':0, 'dev_model_seed_threshold_bin_or_checkpoint_selection':False,
            'final_Val_Test_original_pixels_or_new_nonKnown_semantics_read':False,
            'foreground_Existing_gate_full_deployment_calibration_or_fair_latency_coverage_claimed':False,
            'full_semantic_truth_Hscore_CT_joint_Pred_final_or_Goal_accepted':False,
            'resources':{'peak_guarded_RSS_bytes':peak, 'wall_seconds_excluding_network':time.monotonic()-start,
                         'training_GPU_hours':0, 'CUDA_initialized':torch.cuda.is_initialized()}}
        atomic_json(out/'result.json', result)
        print(json.dumps({'status':result['status'], 'control':args.control, 'calendar_seed':args.seed}), flush=True)
    except Exception as error:
        atomic_json(out/'failure.json', {'exception_type':type(error).__name__, 'error':str(error), 'complete':False}); raise


def parser():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit', required=True); p.add_argument('--git-proxy', default='')
    p.add_argument('--control', choices=('calendar_matched', 'full_policy'), required=True)
    p.add_argument('--seed', type=int, choices=(1027, 1028, 1029)); p.add_argument('--worker', action='store_true')
    return p


def main():
    args = parser().parse_args(); cfg = configuration(ROOT, args.control, args.seed)
    safety = json.loads((ROOT/'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker: worker(args, cfg, safety); return 0
    return supervise(args, cfg, safety)


if __name__ == '__main__': sys.exit(main())
