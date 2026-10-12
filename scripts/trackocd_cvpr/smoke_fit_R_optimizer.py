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
BENCH_CONFIG = 'configs/trackocd_cvpr/fit_R_batched_qualification_v1.json'
SCIENTIFIC_CONFIG = 'configs/trackocd_cvpr/R_GT_scientific_window_v1.json'
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
        '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy,
        '--config', args.config] + (['--variant', args.variant, '--seed', str(args.seed)]
                                  if args.config == SCIENTIFIC_CONFIG else []), cwd=ROOT)
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
    benchmark = cfg.get('batched_qualification_only', False)
    scientific = cfg.get('scientific_R_trial', False)
    if (authority['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or cfg['status'] != ('REGISTERED_FIXED_R_SCIENTIFIC_TRIAL_EXECUTION_DEV_QUALIFICATION_PENDING' if scientific else
                                 'REGISTERED_NO_OPTIMIZER_BATCH_QUALIFICATION' if benchmark else 'REGISTERED_OPTIMIZER_PLUMBING_SOURCE_CACHE_COMPLETION_REQUIRED')
            or cfg['execution_device'] != 'CPU' or cfg['optimizer_steps'] != (200 if scientific else 0 if benchmark else 10)
            or cfg['seed'] not in ([1027, 1028, 1029] if scientific else [1027])
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
        'scope': ('Fixed seeded R trial on fitKnown only; endpoint200, no dev/final selection' if scientific else
                  'Actual fit/checkpoint batch/reference numeric+gradient/timing qualification, zero optimizer' if benchmark else
                  'Ten fit5 Known optimizer steps on complete GT-control feature cache, no dev/final or scientific acceptance')})
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
        if scientific:
            from src.trackocd_cvpr.fit_R_training import scientific_R_steps
            from src.trackocd_cvpr.batched_evidence import TimelessStaticEvidence
            if cfg['variant'].startswith('timeless'):
                torch.manual_seed(cfg['seed']); model = TimelessStaticEvidence().train()
            before = digest(model)
            def save_checkpoint(current, optimizer, step, rng):
                path = out / ('private_step' + str(step) + '_checkpoint.pt')
                if path.exists(): raise ValueError('Preserve every checkpoint')
                temporary = out / (path.name + '.partial')
                with temporary.open('xb') as handle:
                    torch.save({'model_state': current.state_dict(), 'optimizer_state': optimizer.state_dict(),
                        'step': step, 'variant': cfg['variant'], 'seed': cfg['seed'], 'torch_rng_state': torch.get_rng_state(),
                        'numpy_rng_state': rng, 'fit_GT_control_not_Pred': True}, handle)
                    handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, path)
            def progress(row, curve):
                atomic_json(out / 'progress.json', row); atomic_json(out / 'loss_curve.json', curve)
            curve = scientific_R_steps(model, tracks, cfg, cfg['variant'], cfg['seed'], guard, progress, save_checkpoint)
            after = digest(model); endpoint = out / 'private_step200_checkpoint.pt'
            if after == before or torch.cuda.is_initialized(): raise ValueError('Real CPU fit update required')
            restored = torch.load(endpoint, map_location='cpu', weights_only=True)
            verifier = TimelessStaticEvidence() if cfg['variant'].startswith('timeless') else RawPreservingEvidence()
            verifier.load_state_dict(restored['model_state'], strict=True)
            if digest(verifier) != after: raise ValueError('Endpoint model reopen differs')
            guard()
            result = {'status': 'COMPLETED_FIXED_R_FIT_TRIAL_NOT_DEV_VALIDATED_OR_GOAL_ACCEPTED',
                'live_remote_registration_proof': proof, 'variant': cfg['variant'], 'seed': cfg['seed'],
                'completed_optimizer_steps': len(curve), 'label_and_physical_scope': label_proof,
                'model_parameter_sha256_before': before, 'model_parameter_sha256_after': after,
                'endpoint': {'bytes': endpoint.stat().st_size, 'sha256': sha256_file(endpoint), 'reopen_exact': True},
                'loss_curve': curve, 'checkpoint_selection': 'Fixed200 endpoint',
                'dev_final_Val_Test_targets_or_pixels_used': False, 'old_fitted_adapter_initialization': False,
                'all3seeds_or_distinct_R_C_M_mechanisms_or_Goal_A_B_C_accepted': False,
                'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_network': time.monotonic() - started,
                              'training_GPU_hours': 0, 'CUDA_initialized': False}}
            atomic_json(out / 'result.json', result); print(json.dumps(result), flush=True); return
        if benchmark:
            import gc
            from src.trackocd_cvpr.fit_R_training import support_groups
            from src.trackocd_cvpr.exact_support_reduction import reencode_exact_supports
            from src.trackocd_cvpr.batched_evidence import reencode_batched_supports
            checkpoint = ROOT / cfg['private_checkpoint']
            if checkpoint.is_symlink() or sha256_file(checkpoint) != cfg['private_checkpoint_sha256']:
                raise ValueError('Exact immutable prior step10 checkpoint')
            restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(restored['model_state'], strict=True)
            before = digest(model)
            query = next(t for t in tracks if t.loss_target is not None)
            groups = support_groups(tracks, query.video, cfg['class_count'])
            probe = torch.linspace(-1, 1, 768)
            flags = torch.ones(len(groups)); bindings = torch.ones(len(groups))
            guard(); tick = time.monotonic()
            reference = reencode_exact_supports(model, groups, flags, bindings)
            reference[1].mul(probe).sum().backward()
            reference_seconds = time.monotonic() - tick
            reference_values = [x.detach().clone() for x in reference[:3]]
            gradients = {k: p.grad.clone() for k, p in model.named_parameters()}
            del reference; model.zero_grad(set_to_none=True); gc.collect(); guard()
            tick = time.monotonic()
            batched = reencode_batched_supports(model, groups, flags, bindings)
            batched[1].mul(probe).sum().backward()
            batched_seconds = time.monotonic() - tick
            error = max(float((a - b).abs().max()) for a, b in zip(reference_values, batched[:3]))
            if error > cfg['max_forward_absolute_error']:
                raise ValueError('Preregistered fullsupport forward/statistic parity')
            grad_error = 0
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter.grad, gradients[name],
                    atol=cfg['parameter_gradient_atol'], rtol=cfg['parameter_gradient_rtol'])
                grad_error = max(grad_error, float((parameter.grad - gradients[name]).abs().max()))
            if digest(model) != before or torch.cuda.is_initialized():
                raise ValueError('No parameter update or CUDA allocation allowed')
            guard()
            result = {'status': 'VERIFIED_ACTUAL_FIT_CURRENT_R_BATCH_PARITY_AND_TIMING_NOT_OPTIMIZED_OR_SCIENTIFIC_ACCEPTANCE',
                'live_remote_registration_proof': proof, 'label_and_physical_scope': label_proof,
                'support_members': len(batched[3].raw), 'support_observations': sum(len(p.quality) for g in groups for p in g),
                'forward_and_statistics_max_absolute_error': error, 'parameter_gradients_max_absolute_error': grad_error,
                'reference_forward_backward_seconds': reference_seconds, 'batched_forward_backward_seconds': batched_seconds,
                'single_pair_timing_ratio_not_general_performance_claim': reference_seconds / batched_seconds,
                'all_current_R_support_gradients_retained': True, 'model_parameter_sha256_unchanged': before,
                'optimizer_steps': 0, 'dev_final_labels_pixels_originalGT_DINOweights_read': False,
                'private_checkpoint_weights_uploaded': False, 'trained_scientific_mechanism_or_Goal_A_B_C_accepted': False,
                'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_network': time.monotonic() - started,
                              'training_GPU_hours': 0, 'CUDA_initialized': False}}
            atomic_json(out / 'result.json', result); print(json.dumps(result), flush=True); return
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
    global SOURCES
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True); parser.add_argument('--git-proxy', default='')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--config', choices=(CONFIG, BENCH_CONFIG, SCIENTIFIC_CONFIG), default=CONFIG)
    parser.add_argument('--variant', choices=('temporal_CE', 'timeless_CE', 'temporal_Raw_teacher', 'timeless_Raw_teacher'))
    parser.add_argument('--seed', type=int, choices=(1027, 1028, 1029))
    args = parser.parse_args()
    SOURCES = (args.config, *SOURCES[1:])
    if args.config in (BENCH_CONFIG, SCIENTIFIC_CONFIG):
        SOURCES = (*SOURCES[:-1], 'src/trackocd_cvpr/batched_evidence.py',
            'outputs/trackocd_cvpr/cvpr2/fit_R_optimizer_smoke_v1/result.json', SOURCES[-1])
    cfg = json.loads((ROOT / args.config).read_text()); safety = json.loads((ROOT / SOURCES[-1]).read_text())
    if args.config == SCIENTIFIC_CONFIG:
        if args.variant is None or args.seed is None: raise ValueError('Explicit registered variant and independent training seed')
        window = cfg; template = json.loads((ROOT / BENCH_CONFIG).read_text())
        cfg = {**template, **window, **window['proposed_execution'], 'scientific_R_trial': True, 'batched_qualification_only': False,
               'seed': args.seed, 'variant': args.variant, 'optimizer_steps': window['training']['steps_per_variant_and_seed'],
               'class_count': 5, 'execution_device': 'CPU', 'wall_seconds_ceiling': window['proposed_execution']['job_wall_seconds_ceiling'],
               'rss_bytes_ceiling': window['proposed_execution']['job_RSS_bytes_ceiling'], 'output_bytes_ceiling': 8388608,
               'output_directory': 'outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_' + args.variant + '_s' + str(args.seed)}
        SOURCES = (*SOURCES[:-1], BENCH_CONFIG, 'outputs/trackocd_cvpr/cvpr2/fit_R_batched_qualification_v1/result.json', SOURCES[-1])
    if args.worker: worker(args, cfg, safety); return 0
    return supervise(args, cfg, safety)


if __name__ == '__main__': sys.exit(main())
