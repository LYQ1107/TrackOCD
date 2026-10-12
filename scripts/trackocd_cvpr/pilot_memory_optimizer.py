#!/usr/bin/env python3
"""Fresh registered ten-step M pilot under an owned CPU supervisor."""
import argparse
from dataclasses import asdict
import hashlib
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
from scripts.trackocd_cvpr.pilot_candidate_optimizer import SOURCES as C_SOURCES

CONFIG = 'configs/trackocd_cvpr/memory_optimizer_pilot_v1.json'
SOURCES = (*C_SOURCES, CONFIG, 'scripts/trackocd_cvpr/pilot_memory_optimizer.py',
    'src/trackocd_cvpr/member_reliability.py', 'src/trackocd_cvpr/reliability_support_bridge.py',
    'src/trackocd_cvpr/committed_support_bridge.py', 'src/trackocd_cvpr/versioned_reliability_memory.py',
    'src/trackocd_cvpr/reliability_member_control.py', 'src/trackocd_cvpr/reliability_training_loss.py',
    'src/trackocd_cvpr/memory_fit_pilot.py', 'src/trackocd_cvpr/candidate_scientific_training.py',
    'src/trackocd_cvpr/C_development_rollout.py', 'src/trackocd_cvpr/R_development_evaluation.py',
    'src/trackocd_cvpr/R_development_metrics.py', 'src/trackocd_cvpr/causal_category_audit.py')


def validate_configuration(cfg, authority):
    if (authority['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or cfg['status'] != 'REGISTERED_TEN_M_STEPS_FULL_POLICY_PREDICTION_OWNED_HISTORY_REQUIRED'
            or cfg['optimizer_steps'] != 10 or cfg['seed'] != 1027 or cfg['execution_device'] != 'CPU'
            or cfg['K_per_family'] != 8 or cfg['full_scientific_fit_automatic_start'] is not False
            or cfg['dev_final_Val_Test_new_nonKnown_labels_or_pixels']):
        raise ValueError('Exact authorized bounded ten-step M pilot only')


def supervise(args, cfg, safety):
    out = ROOT / cfg['output_directory']; receipt = out.with_name(out.name + '_supervisor_attempt_1.json')
    if out.exists() or receipt.exists(): raise ValueError('Preserve all previous attempts')
    started, peak, threads, reason = time.monotonic(), 0, 0, None
    if resources_ok(cfg, safety, ROOT, started, 0): raise RuntimeError('Pre-spawn M resource guard')
    child = subprocess.Popen([sys.executable, '-B', '-X', 'pycache_prefix=' + str(out / '_fresh_no_write_cache'),
        str(ROOT / 'scripts/trackocd_cvpr/pilot_memory_optimizer.py'), '--worker',
        '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy], cwd=ROOT)
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
    steps = json.loads((out / 'progress.json').read_text())['step'] if (out / 'progress.json').exists() else 0
    atomic_json(receipt, {'owned_worker_exit_code': code, 'guard_stop_reason': reason,
        'successful_owned_execution': code == 0 and reason is None, 'only_new_owned_Popen_worker_managed': True,
        'last_persisted_completed_optimizer_steps': steps, 'peak_sampled_owned_worker_RSS_bytes': peak,
        'peak_sampled_worker_threads': threads, 'wall_seconds_including_imports_and_network': time.monotonic() - started,
        'training_GPU_hours': 0, 'execution_device': 'CPU'})
    return 0 if code == 0 and reason is None else 1


def worker(args, cfg, safety):
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(ROOT / cfg['output_directory'] / '_fresh_no_write_cache'):
        raise ValueError('Fresh no-write owned M worker required')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for path in SOURCES:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Actual M pilot source/cache/authority registration required')
    validate_configuration(cfg, json.loads((ROOT / cfg['approval']).read_text()))
    qualification = ROOT / 'outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1'
    terminal = json.loads(qualification.with_name(qualification.name + '_supervisor_attempt_1.json').read_text())
    if (sha256_file(qualification / 'result.json') != cfg['policy_input_qualification_sha256']
            or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None):
        raise ValueError('Actual full policy input qualification required before private GT')
    out = ROOT / cfg['output_directory']
    if out.exists(): raise ValueError('No output overwrite')
    out.mkdir(parents=True); started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, process_memory()['guarded_process_RSS_bytes'])
        reason = resources_ok(cfg, safety, ROOT, started, peak)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes + size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned M pilot guard:' + str(reason))
    atomic_json(out / 'launch.json', {'live_remote_registration_proof': proof,
        'source_sha256': {p: sha256_file(ROOT / p) for p in SOURCES},
        'scope': 'Ten actual M updates on full prediction-owned policy history, no dev/scientific acceptance'})
    try:
        import torch
        from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
        from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks, support_groups
        from src.trackocd_cvpr.fit_loss_supervision import project_fit_loss_targets
        from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
        from src.trackocd_cvpr.batched_evidence import reencode_batched_supports
        from src.trackocd_cvpr.memory_fit_pilot import fit_memory_pilot
        from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
        from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
        torch.set_num_threads(1)
        def barrier(event, values):
            if event == 'socket.connect': raise PermissionError('No network after source registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                path = Path(os.fsdecode(values[0])).resolve()
                if path.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (
                        path.suffix in ('.pt', '.pth') and path.parent != out):
                    raise PermissionError('Qualified cache/own endpoint only, no original data or external weights')
        sys.addaudithook(barrier)
        guard(); videos, targets, policy_proof = load_completed_policy_videos(ROOT, cfg, guard)
        fit_tracks, fit_proof = load_completed_fit_tracks(ROOT, cfg, guard)
        bindings, _, _ = project_fit_loss_targets(ROOT / cfg['private_assignment'], ROOT / cfg['private_Known_support'])
        if bindings != targets.fit_bindings: raise ValueError('Exact fixed Known prior binding order')
        torch.manual_seed(cfg['seed']); raw_model = RawPreservingEvidence().eval().requires_grad_(False)
        with torch.inference_mode():
            raw, _, _, _ = reencode_batched_supports(raw_model, support_groups(fit_tracks, -1, 5), torch.ones(5), torch.ones(5))
        known = {i: row.numpy().copy() for i, row in enumerate(raw)}
        model, optimizer, summary, sealed, history, versions, commits = fit_memory_pilot(videos, targets, known, cfg, guard,
            lambda row, curve: (atomic_json(out / 'progress.json', row), atomic_json(out / 'loss_curve.json', curve)))
        endpoint = out / 'private_step10_checkpoint.pt'; temporary = out / 'private_step10_checkpoint.partial'
        with temporary.open('xb') as handle:
            torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'step': 10, 'seed': cfg['seed'], 'torch_rng_state': torch.get_rng_state(),
                'not_automatic_causal_bank_resume': True}, handle)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, endpoint)
        verifier = MemberReliabilityMLP()
        verifier.load_state_dict(torch.load(endpoint, map_location='cpu', weights_only=True)['model_state'], strict=True)
        if digest(verifier) != digest(model) or torch.cuda.is_initialized(): raise ValueError('Exact fixed10 CPU M endpoint')
        version_rows = []
        for version in versions:
            row = asdict(version); prototype = row.pop('prototype_float64_bytes')
            row['prototype_sha256'] = hashlib.sha256(prototype).hexdigest() if prototype is not None else None
            version_rows.append(row)
        history_path = out / 'private_prediction_history.json'
        atomic_json(history_path, {'versions': version_rows, 'actual_bank_commits': [asdict(c) for c in commits],
            'anonymous_commit_labels': [asdict(r) for r in history],
            'actual_predictions': [{'video_id': video, 'frame_index': r.frame_index,
                'routing_groups': r.routing_groups, 'chosen_preframe_versions': chosen,
                'events': [{'action': e.action, 'category_id': e.category_id, 'token': e.token,
                            'ambiguity_reason': e.ambiguity_reason} for e in r.events]} for video, r, chosen in sealed]})
        guard()
        result = {'status': 'COMPLETED_TEN_M_STEPS_FULL_POLICY_ROLLOUT_NOT_SCIENTIFIC_M_OR_GOAL_ACCEPTANCE',
            'live_remote_registration_proof': proof, 'seed': cfg['seed'], 'summary': summary,
            'policy_scope': policy_proof, 'offline_fit_prior_scope': fit_proof,
            'endpoint': {'bytes': endpoint.stat().st_size, 'sha256': sha256_file(endpoint), 'reopen_exact': True},
            'private_prediction_history': {'bytes': history_path.stat().st_size, 'sha256': sha256_file(history_path)},
            'checkpoint_selection': 'Fixed10 endpoint', 'old_R_C_M_or_pilot_warmstart': False,
            'dev_final_Val_Test_or_new_nonKnown_labels_used': False,
            'three_seed_M_strict_CT_Pred_or_Goal_accepted': False,
            'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_network': time.monotonic() - started,
                          'training_GPU_hours': 0, 'CUDA_initialized': False}}
        atomic_json(out / 'result.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'summary'}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False}); raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True); parser.add_argument('--git-proxy', default='')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(); cfg = json.loads((ROOT / CONFIG).read_text())
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker: worker(args, cfg, safety); return 0
    return supervise(args, cfg, safety)


if __name__ == '__main__': sys.exit(main())
