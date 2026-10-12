#!/usr/bin/env python3
"""One owned CPU worker, ten real fit-only R steps after complete cache proof."""
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

CONFIG = 'configs/trackocd_cvpr/fit_R_optimizer_smoke_v1.json'
SOURCES = (CONFIG, 'scripts/trackocd_cvpr/smoke_fit_R_optimizer.py',
    'src/trackocd_cvpr/fit_R_training.py', 'src/trackocd_cvpr/fit_loss_supervision.py',
    'src/trackocd_cvpr/exact_support_reduction.py', 'src/trackocd_cvpr/differentiable_support_bank.py',
    'src/trackocd_cvpr/raw_preserving_evidence.py', 'src/trackocd_cvpr/GT_video_cache.py',
    'src/trackocd_core/features.py', 'src/trackocd_cvpr/scoped_train_census.py',
    'src/trackocd_cvpr/metadata_only.py', 'src/trackocd_cvpr/remote_registration.py', 'src/trackocd_v2/io.py',
    'scripts/trackocd_cvpr/smoke_selected_frozen_encoder.py',
    'scripts/trackocd_cvpr/audit_joint_known_protocol.py', 'scripts/trackocd_cvpr/audit_train_metadata_containment.py',
    'configs/trackocd_cvpr/formal_research_scope_approval_v1.json', 'configs/trackocd_cvpr/frozen_global_data_GT_control_v1.json',
    'outputs/trackocd_cvpr/cvpr2/fit_GT_video_cache_v1/result.json',
    'outputs/trackocd_cvpr/cvpr2/fit_GT_video_cache_v1_supervisor_attempt_1.json',
    'configs/trackocd_cvpr/autonomous_research.json')


def supervise(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    receipt = out.with_name(out.name + '_supervisor_attempt_1.json')
    if out.exists() or receipt.exists(): raise ValueError('Preserve every previous owned attempt')
    started, peak, threads, reason = time.monotonic(), 0, 0, None
    if resources_ok(cfg, safety, ROOT, started, 0): raise RuntimeError('Pre-spawn guard')
    child = subprocess.Popen([sys.executable, '-B', '-X', 'pycache_prefix=' + str(out / '_fresh_no_write_cache'),
        str(ROOT / 'scripts/trackocd_cvpr/smoke_fit_R_optimizer.py'), '--worker',
        '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy], cwd=ROOT)
    try:
        while child.poll() is None:
            status_path = Path(f'/proc/{child.pid}/status')
            try:
                status = status_path.read_text(); rollup = Path(f'/proc/{child.pid}/smaps_rollup').read_text()
                peak = max(peak, process_memory(status, rollup)['guarded_process_RSS_bytes'])
                fields = {line.split(':', 1)[0]: line.split(':', 1)[1] for line in status.splitlines() if ':' in line}
                threads = max(threads, int(fields['Threads'].strip()))
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                if child.poll() is not None: break
                try: terminal = kernel_terminal_status(status_path.read_text())
                except (FileNotFoundError, ProcessLookupError): terminal = False
                if terminal: child.wait(timeout=1); break
                if child.poll() is not None: break
                if time.monotonic() - started > 2: reason = 'owned_memory_counter_unavailable'; child.kill(); break
                time.sleep(0.2); continue
            reason = resources_ok(cfg, safety, ROOT, started, peak)
            if out.exists() and sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > cfg['output_bytes_ceiling']:
                reason = 'owned_output_size'
            if reason:
                if child.poll() is None: child.kill()  # only this newly created Popen child
                break
            time.sleep(0.2)
        code = child.wait(timeout=10)
    except BaseException:
        if child.poll() is None: child.kill()
        child.wait(timeout=10); raise
    steps = 0
    if (out / 'progress.json').exists(): steps = json.loads((out / 'progress.json').read_text())['step']
    atomic_json(receipt, {'owned_worker_exit_code': code, 'guard_stop_reason': reason,
        'successful_owned_execution': code == 0 and reason is None, 'only_new_owned_Popen_worker_managed': True,
        'last_persisted_completed_optimizer_steps': steps, 'peak_sampled_owned_worker_RSS_bytes': peak,
        'peak_sampled_worker_threads': threads, 'wall_seconds_including_imports_and_network': time.monotonic() - started,
        'training_GPU_hours': 0, 'execution_device': 'CPU'})
    return 0 if code == 0 and reason is None else 1


def worker(args, cfg, safety):
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(ROOT / cfg['output_directory'] / '_fresh_no_write_cache'):
        raise ValueError('Fresh no-write CPU worker required')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for p in SOURCES:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + p], cwd=ROOT) != (ROOT / p).read_bytes():
            raise ValueError('Source or completed cache receipt not truly registered')
    authority = json.loads((ROOT / cfg['approval']).read_text())
    if (authority['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or cfg['status'] != 'REGISTERED_OPTIMIZER_PLUMBING_SOURCE_CACHE_COMPLETION_REQUIRED'
            or cfg['execution_device'] != 'CPU' or cfg['optimizer_steps'] != 10 or cfg['seed'] != 1027
            or cfg['class_count'] != 5 or cfg['full_scientific_fit_automatic_start'] is not False):
        raise ValueError('Exact new authority / bounded first optimizer smoke only')
    out = ROOT / cfg['output_directory']
    if out.exists(): raise ValueError('Preserve old attempts')
    out.mkdir(parents=True)
    started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, process_memory()['guarded_process_RSS_bytes'])
        reason = resources_ok(cfg, safety, ROOT, started, peak)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes + size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned CPU optimization guard:' + str(reason))
    atomic_json(out / 'launch.json', {'preregistration_commit': args.preregistration_commit,
        'live_remote_registration_proof': proof, 'source_sha256': {p: sha256_file(ROOT / p) for p in SOURCES},
        'scope': 'Ten fit5 Known optimizer steps on complete GT-control feature cache, no dev/final or scientific acceptance'})
    try:
        import torch
        from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks, bounded_fit_steps
        from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
        from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
        torch.set_num_threads(1); torch.manual_seed(cfg['seed'])
        def barrier(event, values):
            if event == 'socket.connect': raise PermissionError('No network after registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                p = Path(os.fsdecode(values[0])).resolve()
                if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')):
                    raise PermissionError('Cache-only fit, no original annotations or pixels')
        sys.addaudithook(barrier)
        guard(); tracks, label_proof = load_completed_fit_tracks(ROOT, cfg, guard)
        model = RawPreservingEvidence().train()
        before = digest(model)
        if sum(p.numel() for p in model.parameters()) != 50113:
            raise ValueError('Fresh registered R parameter budget')
        optimizer, curve, numpy_rng = bounded_fit_steps(model, tracks, cfg, guard,
            lambda row: atomic_json(out / 'progress.json', row))
        after = digest(model)
        if after == before or torch.cuda.is_initialized():
            raise ValueError('Actual fresh-R updates and CPU-only execution required')
        checkpoint = out / 'private_step10_R_checkpoint.pt'
        with checkpoint.open('xb') as handle:
            torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'optimizer_steps': len(curve), 'seed': cfg['seed'], 'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': numpy_rng, 'GT_control_only_not_Pred_or_scientific_acceptance': True}, handle)
            handle.flush(); os.fsync(handle.fileno())
        guard()
        with checkpoint.open('rb') as handle:
            restored = torch.load(handle, map_location='cpu', weights_only=True)
        verifier = RawPreservingEvidence(); verifier.load_state_dict(restored['model_state'], strict=True)
        if digest(verifier) != after or restored['optimizer_steps'] != 10 or 'optimizer_state' not in restored:
            raise ValueError('Private endpoint checkpoint reopen differs')
        result = {'status': 'VERIFIED_TEN_REAL_FIT_KNOWN_CPU_R_OPTIMIZER_STEPS_NOT_SCIENTIFIC_ACCEPTANCE',
            'live_remote_registration_proof': proof, 'seed': cfg['seed'], 'completed_optimizer_steps': len(curve),
            'label_and_physical_scope': label_proof, 'loss_curve': curve,
            'model_parameter_sha256_before': before, 'model_parameter_sha256_after': after,
            'private_checkpoint': {'bytes': checkpoint.stat().st_size, 'sha256': sha256_file(checkpoint), 'reopen_exact': True},
            'numeric_model_inputs_contain_labels_roles_IDs_or_futurelength': False,
            'queryself_or_samevideo_in_offline_Known_support': False,
            'development_final_pixels_or_labels_originalGT_or_DINOweights_loaded': False,
            'offline_GT_training_augmentation_not_frame_online_controller': True,
            'trained_scientific_seeds_or_R_C_M_mechanisms_or_Goal_A_B_C_accepted': False,
            'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_network': time.monotonic() - started,
                          'training_GPU_hours': 0, 'CUDA_initialized': False, 'execution_device': 'CPU'}}
        guard(); atomic_json(out / 'result.json', result); print(json.dumps(result), flush=True)
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
