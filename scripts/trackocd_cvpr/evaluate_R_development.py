#!/usr/bin/env python3
"""Registered CPU dev-only GT controls; no optimizer or winner selection."""
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

CONFIG = 'configs/trackocd_cvpr/R_development_first_pair_v1.json'
SOURCES = (CONFIG, 'scripts/trackocd_cvpr/evaluate_R_development.py',
    'src/trackocd_cvpr/R_development_evaluation.py', 'src/trackocd_cvpr/R_development_metrics.py',
    'src/trackocd_cvpr/fit_R_training.py', 'src/trackocd_cvpr/fit_loss_supervision.py',
    'src/trackocd_cvpr/GT_video_cache.py', 'src/trackocd_cvpr/batched_evidence.py',
    'src/trackocd_cvpr/raw_preserving_evidence.py', 'src/trackocd_cvpr/differentiable_support_bank.py',
    'src/trackocd_cvpr/exact_support_reduction.py', 'src/trackocd_core/features.py',
    'src/trackocd_cvpr/scoped_train_census.py', 'src/trackocd_cvpr/metadata_only.py',
    'src/trackocd_cvpr/remote_registration.py', 'src/trackocd_v2/io.py',
    'scripts/trackocd_cvpr/smoke_selected_frozen_encoder.py',
    'scripts/trackocd_cvpr/audit_joint_known_protocol.py', 'scripts/trackocd_cvpr/audit_train_metadata_containment.py',
    'configs/trackocd_cvpr/R_GT_scientific_window_v1.json', 'configs/trackocd_cvpr/R_GT_development_evaluation_v1.json',
    'configs/trackocd_cvpr/formal_research_scope_approval_v1.json', 'configs/trackocd_cvpr/frozen_global_data_GT_control_v1.json',
    'outputs/trackocd_cvpr/cvpr2/fit_GT_video_cache_v1/result.json',
    'outputs/trackocd_cvpr/cvpr2/fit_GT_video_cache_v1_supervisor_attempt_1.json',
    'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v3_resource_retry/result.json',
    'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v3_resource_retry_supervisor_attempt_1.json',
    'outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_temporal_Raw_teacher_s1027/result.json',
    'outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_timeless_Raw_teacher_s1027/result.json',
    'configs/trackocd_cvpr/autonomous_research.json')


def supervise(args, cfg, safety):
    out = ROOT / cfg['output_directory']; receipt = out.with_name(out.name + '_supervisor_attempt_1.json')
    if out.exists() or receipt.exists(): raise ValueError('Preserve every previous attempt')
    started, peak, threads, reason = time.monotonic(), 0, 0, None
    if resources_ok(cfg, safety, ROOT, started, 0): raise RuntimeError('Pre-spawn resource guard')
    child = subprocess.Popen([sys.executable, '-B', '-X', 'pycache_prefix=' + str(out / '_fresh_no_write_cache'),
        str(ROOT / 'scripts/trackocd_cvpr/evaluate_R_development.py'), '--worker',
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
                if child.poll() is None: child.kill()
                break
            time.sleep(0.2)
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
        raise ValueError('Fresh no-write worker')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for path in SOURCES:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Source / complete cache / fixedendpoint receipt not truly registered')
    permission = json.loads((ROOT / cfg['approval']).read_text())
    if (permission['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or cfg['models'] != ['temporal_Raw_teacher', 'timeless_Raw_teacher'] or cfg['seed'] != 1027
            or cfg['checkpoint_step'] != 200 or cfg['select_model_or_checkpoint'] is not False):
        raise ValueError('Fixed firstpair dev analysis, never a winning-seed selection')
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
            raise RuntimeError('Owned development evaluator guard:' + str(reason))
    atomic_json(out / 'launch.json', {'live_remote_registration_proof': proof,
        'source_sha256': {p: sha256_file(ROOT / p) for p in SOURCES}, 'scope': 'Read-only firstpair GT Train-development, no fitting/selection/final'})
    try:
        import torch
        from src.trackocd_cvpr.R_development_evaluation import (load_completed_dev_tracks, known_prototypes,
            embed_complete_role, evaluate_prefix_embeddings, load_completed_fit_tracks)
        from src.trackocd_cvpr.fit_loss_supervision import project_fit_loss_targets
        from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
        from src.trackocd_cvpr.batched_evidence import TimelessStaticEvidence
        from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
        torch.set_num_threads(1); torch.manual_seed(1027)
        def barrier(event, values):
            if event == 'socket.connect': raise PermissionError('No network after registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                p = Path(os.fsdecode(values[0])).resolve()
                if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')):
                    raise PermissionError('No original annotation / dataset pixels, qualified cache only')
        sys.addaudithook(barrier)
        guard(); dev_tracks, labels, label_proof = load_completed_dev_tracks(ROOT, cfg, guard)
        fit_tracks, fit_proof = load_completed_fit_tracks(ROOT, cfg, guard)
        bindings, _, _ = project_fit_loss_targets(ROOT / cfg['private_assignment'], ROOT / cfg['private_Known_support'])
        table = {}; raw_reference = {}; raw_vectors = {}
        raw_model = RawPreservingEvidence().eval().requires_grad_(False)
        raw_prototypes, _ = known_prototypes(raw_model, fit_tracks)
        for cap in cfg['prefixes']:
            raw, _, actual = embed_complete_role(raw_model, dev_tracks, cap, guard)
            raw_vectors[str(cap)] = (raw, actual)
        for variant in cfg['models']:
            directory = ROOT / ('outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_' + variant + '_s1027')
            receipt = json.loads((directory / 'result.json').read_text())
            terminal = json.loads(directory.with_name(directory.name + '_supervisor_attempt_1.json').read_text())
            if (receipt['completed_optimizer_steps'] != 200 or receipt['variant'] != variant or receipt['seed'] != 1027
                    or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
                    or receipt['dev_final_Val_Test_targets_or_pixels_used'] or receipt['old_fitted_adapter_initialization']):
                raise ValueError('Fixed actualfit endpoint, not best dev/loss checkpoint')
            checkpoint = directory / 'private_step200_checkpoint.pt'
            if checkpoint.is_symlink() or sha256_file(checkpoint) != receipt['endpoint']['sha256']:
                raise ValueError('Endpoint privateweight fingerprint')
            model = TimelessStaticEvidence() if variant.startswith('timeless') else RawPreservingEvidence()
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state'], strict=True)
            model.eval().requires_grad_(False); before = digest(model)
            _, adapted_prototypes = known_prototypes(model, fit_tracks)
            rows = {}
            for cap in cfg['prefixes']:
                _, adapted, actual = embed_complete_role(model, dev_tracks, cap, guard)
                key = str(cap)
                raw, raw_actual = raw_vectors[key]
                if actual != raw_actual: raise ValueError('Exact shared observed-prefix footprint')
                current = evaluate_prefix_embeddings(raw, adapted, dev_tracks, labels, bindings,
                                                      raw_prototypes, adapted_prototypes, actual)
                if key in raw_reference:
                    if current['Raw'] != raw_reference[key]:
                        raise ValueError('Independent frozenRaw reference differs across compared models')
                else: raw_reference[key] = current['Raw']
                rows[key] = current
                atomic_json(out / 'progress.json', {'completed_variant': variant, 'completed_prefix': key,
                    'physical_tracks': len(dev_tracks), 'no_training_or_winner_selection': True})
            if digest(model) != before or torch.cuda.is_initialized(): raise ValueError('Read-only CPU model state unchanged')
            table[variant] = rows
        guard()
        result = {'status': 'MEASURED_FIRSTPAIR_GT_TRAIN_DEVELOPMENT_NOT_ALL_SEEDS_PRED_OR_GOAL_ACCEPTANCE',
            'live_remote_registration_proof': proof, 'development_scope': label_proof, 'fit_prototype_scope': fit_proof,
            'seed': 1027, 'checkpoint_step': 200, 'results': table, 'Raw_reference': raw_reference,
            'optimizer_steps': 0, 'model_or_checkpoint_selection': False,
            'final_Val_Test_or_new_nonKnown_semantic_values_read': False,
            'complete_semantic_truth_Hscore_purity_or_R_C_M_Goal_accepted': False,
            'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_network': time.monotonic() - started,
                          'training_GPU_hours': 0, 'CUDA_initialized': False}}
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
