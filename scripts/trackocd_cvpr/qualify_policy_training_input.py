#!/usr/bin/env python3
"""Fresh-registered owned CPU policy input proof; zero optimizer/encoder."""
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
from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok, kernel_terminal_status
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory

CONFIG = 'configs/trackocd_cvpr/policy_training_input_qualification_v1.json'
SOURCES = (CONFIG, 'scripts/trackocd_cvpr/qualify_policy_training_input.py',
    'src/trackocd_cvpr/policy_training_data.py', 'src/trackocd_cvpr/causal_observation_router.py',
    'src/trackocd_cvpr/GT_video_cache.py', 'src/trackocd_core/features.py',
    'src/trackocd_cvpr/scoped_train_census.py', 'src/trackocd_cvpr/metadata_only.py',
    'src/trackocd_cvpr/remote_registration.py', 'src/trackocd_v2/io.py',
    'scripts/trackocd_cvpr/audit_joint_known_protocol.py', 'scripts/trackocd_cvpr/audit_train_metadata_containment.py',
    'configs/trackocd_cvpr/formal_research_scope_approval_v1.json',
    'configs/trackocd_cvpr/frozen_global_data_GT_control_v1.json',
    'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v3_resource_retry/result.json',
    'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v3_resource_retry_supervisor_attempt_1.json',
    'configs/trackocd_cvpr/autonomous_research.json')


def supervise(args, cfg, safety, *, worker_script='scripts/trackocd_cvpr/qualify_policy_training_input.py'):
    if worker_script not in ('scripts/trackocd_cvpr/qualify_policy_training_input.py',
                             'scripts/trackocd_cvpr/pilot_candidate_optimizer.py',
                             'scripts/trackocd_cvpr/evaluate_C_development.py'):
        raise ValueError('Explicit registered owned CPU worker only')
    out = ROOT / cfg['output_directory']; receipt = out.with_name(out.name + '_supervisor_attempt_1.json')
    if out.exists() or receipt.exists(): raise ValueError('Preserve all previous attempts')
    started, peak, threads, reason = time.monotonic(), 0, 0, None
    if resources_ok(cfg, safety, ROOT, started, 0): raise RuntimeError('Pre-spawn resource guard')
    extra = []
    if worker_script == 'scripts/trackocd_cvpr/pilot_candidate_optimizer.py':
        extra = ['--config', args.config]
        if args.config == 'configs/trackocd_cvpr/C_GT_scientific_window_v1.json':
            extra += ['--variant', args.variant, '--seed', str(args.seed)]
    elif worker_script == 'scripts/trackocd_cvpr/evaluate_C_development.py':
        extra = ['--model', args.model, '--seed', str(args.seed)]
        if getattr(args, 'recovery_attempt', None) is not None:
            extra += ['--recovery-attempt', args.recovery_attempt]
    child = subprocess.Popen([sys.executable, '-B', '-X', 'pycache_prefix=' + str(out / '_fresh_no_write_cache'),
        str(ROOT / worker_script), '--worker',
        '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy] + extra, cwd=ROOT)
    try:
        while child.poll() is None:
            status_path = Path(f'/proc/{child.pid}/status')
            try:
                status = status_path.read_text(); rollup = Path(f'/proc/{child.pid}/smaps_rollup').read_text()
                peak = max(peak, process_memory(status, rollup)['guarded_process_RSS_bytes'])
                fields = {l.split(':', 1)[0]: l.split(':', 1)[1] for l in status.splitlines() if ':' in l}
                threads = max(threads, int(fields['Threads'].strip()))
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                if child.poll() is not None: break
                try: terminal = kernel_terminal_status(status_path.read_text())
                except (FileNotFoundError, ProcessLookupError): terminal = False
                if terminal: child.wait(timeout=1); break
                if child.poll() is not None: break
                if time.monotonic() - started > 2: reason = 'owned_memory_counter_unavailable'; child.kill(); break
                time.sleep(.2); continue
            reason = resources_ok(cfg, safety, ROOT, started, peak)
            if out.exists() and sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > cfg['output_bytes_ceiling']:
                reason = 'owned_output_size'
            if reason:
                if child.poll() is None: child.kill()  # only this fresh owned Popen child
                break
            time.sleep(.2)
        code = child.wait(timeout=10)
    except BaseException:
        if child.poll() is None: child.kill()
        child.wait(timeout=10); raise
    atomic_json(receipt, {'owned_worker_exit_code': code, 'guard_stop_reason': reason,
        'successful_owned_execution': code == 0 and reason is None, 'only_new_owned_Popen_worker_managed': True,
        'peak_sampled_owned_worker_RSS_bytes': peak, 'peak_sampled_worker_threads': threads,
        'wall_seconds_including_imports_and_network': time.monotonic() - started, 'training_GPU_hours': 0})
    return 0 if code == 0 and reason is None else 1


def worker(args, cfg, safety):
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(ROOT / cfg['output_directory'] / '_fresh_no_write_cache'):
        raise ValueError('Fresh no-write worker required')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for path in SOURCES:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Source/cache receipt not truly preregistered')
    authority = json.loads((ROOT / cfg['approval']).read_text())
    if (authority['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or cfg['status'] != 'REGISTERED_POLICY_INPUT_ONLY_NO_OPTIMIZATION_OR_METHOD_ACCEPTANCE'
            or cfg['optimizer_steps'] != 0 or cfg['execution_device'] != 'CPU'):
        raise ValueError('Exact authorized policy input-only execution')
    out = ROOT / cfg['output_directory']
    if out.exists(): raise ValueError('No overwrite')
    out.mkdir(parents=True); started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, process_memory()['guarded_process_RSS_bytes'])
        reason = resources_ok(cfg, safety, ROOT, started, peak)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes + size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned policy input guard:' + str(reason))
    atomic_json(out / 'launch.json', {'live_remote_registration_proof': proof,
        'source_sha256': {p: sha256_file(ROOT / p) for p in SOURCES}, 'scope': 'Policy input qualification, zero optimization'})
    try:
        from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos, policy_current_frames
        from src.trackocd_cvpr.causal_observation_router import CurrentObservationRouter, visible_bytes
        def barrier(event, values):
            if event == 'socket.connect': raise PermissionError('No network after registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                p = Path(os.fsdecode(values[0])).resolve()
                if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or p.suffix in ('.pth', '.pt'):
                    raise PermissionError('Qualified policy cache only, no original annotations/pixels/weights')
        sys.addaudithook(barrier)
        guard(); videos, targets, data_proof = load_completed_policy_videos(ROOT, cfg, guard)
        frames = empty = observations = groups = 0
        for video, arrays in videos:
            guard(); original = CurrentObservationRouter(); renamed = CurrentObservationRouter()
            routes = {int(t): 'opaque-renamed-' + str(i) for i, t in enumerate(reversed(arrays['track_id']))}
            left = policy_current_frames(arrays); right = policy_current_frames(arrays, routing_bindings=routes)
            for (frame, a), (renamed_frame, b) in zip(left, right):
                guard()
                if frame != renamed_frame: raise ValueError('Exact metadata clock under bijective routing')
                x = original.ingest(frame, a); y = renamed.ingest(frame, reversed(b))
                if tuple(visible_bytes(v) for v in x.model_views) != tuple(visible_bytes(v) for v in y.model_views):
                    raise ValueError('Actual numeric policy prefixes depend on IDs/input order')
                frames += 1; empty += int(not a); observations += len(a); groups += len(x.model_views)
        if (frames, empty, observations) != (374, 1, 1268): raise ValueError('No hidden frame/observation filtering')
        guard()
        result = {'status': 'VERIFIED_COMPLETE_POLICY_INPUT_NOT_TRAINED_C_M_OR_PRED_OR_GOAL_ACCEPTANCE',
            'live_remote_registration_proof': proof, 'policy_scope': data_proof,
            'actual_frame_adapter': {'metadata_frames': frames, 'empty_frames': empty,
                'physical_observations': observations, 'canonical_visible_groups': groups,
                'bijective_ID_rename_and_reversed_delivery_numeric_inputs_equal': True},
            'optimizer_steps': 0, 'model_or_memory_training_or_scientific_acceptance': False,
            'original_annotations_pixels_weights_or_protected_semantics_read': False,
            'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_network': time.monotonic() - started,
                          'training_GPU_hours': 0}}
        atomic_json(out / 'result.json', result); print(json.dumps(result), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False}); raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True); parser.add_argument('--git-proxy', default='')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(); cfg = json.loads((ROOT / CONFIG).read_text()); safety = json.loads((ROOT / SOURCES[-1]).read_text())
    if args.worker: worker(args, cfg, safety); return 0
    return supervise(args, cfg, safety)


if __name__ == '__main__': sys.exit(main())
