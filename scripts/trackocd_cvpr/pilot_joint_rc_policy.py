#!/usr/bin/env python3
"""Registered bounded joint R/C + anonymous-memory Train pilot."""

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

CONFIG = 'configs/trackocd_cvpr/joint_rc_policy_pilot_v1.json'
SOURCES = (
    CONFIG,
    'scripts/trackocd_cvpr/pilot_joint_rc_policy.py',
    'src/trackocd_cvpr/joint_rc_policy_training.py',
    'src/trackocd_cvpr/raw_preserving_evidence.py',
    'src/trackocd_cvpr/candidate_score_contract.py',
    'src/trackocd_cvpr/differentiable_candidate_features.py',
    'src/trackocd_cvpr/differentiable_support_bank.py',
    'src/trackocd_cvpr/batched_evidence.py',
    'src/trackocd_cvpr/atomic_member_control.py',
    'src/trackocd_cvpr/causal_observation_router.py',
    'src/trackocd_cvpr/prefix_member_bank.py',
    'src/trackocd_cvpr/candidate_training_loss.py',
    'src/trackocd_cvpr/policy_training_data.py',
    'src/trackocd_cvpr/fit_R_training.py',
    'src/trackocd_cvpr/fit_loss_supervision.py',
    'src/trackocd_cvpr/GT_video_cache.py',
    'src/trackocd_core/features.py',
    'src/trackocd_cvpr/scoped_train_census.py',
    'src/trackocd_cvpr/remote_registration.py',
    'src/trackocd_v2/io.py',
    'scripts/trackocd_cvpr/smoke_selected_frozen_encoder.py',
    'scripts/trackocd_cvpr/audit_joint_known_protocol.py',
    'scripts/trackocd_cvpr/audit_train_metadata_containment.py',
    'configs/trackocd_cvpr/formal_research_scope_approval_v1.json',
    'configs/trackocd_cvpr/frozen_global_data_GT_control_v1.json',
    'configs/trackocd_cvpr/autonomous_research.json',
    'outputs/trackocd_cvpr/cvpr2/fit_GT_video_cache_v1/result.json',
    'outputs/trackocd_cvpr/cvpr2/fit_GT_video_cache_v1_supervisor_attempt_1.json',
    'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v3_resource_retry/result.json',
    'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v3_resource_retry_supervisor_attempt_1.json',
    'outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1/result.json',
    'outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1_supervisor_attempt_1.json',
)


def _source_gate(args, cfg, safety):
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit,
                                args.git_proxy, allow_public_api=True)
    for path in SOURCES:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Registered source differs from live source: ' + path)
    authority = json.loads((ROOT / cfg['approval']).read_text())
    if (authority['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or cfg['status'] != 'REGISTERED_JOINT_RC_MEMORY_PILOT_NOT_SCIENTIFIC_ACCEPTANCE'
            or cfg['joint_feature_count'] != 14
            or cfg['evidence_parameters'] != 50113 or cfg['scorer_parameters'] != 804
            or cfg['execution_device'] != 'CPU' or cfg['CPU_compute_workers'] != 1
            or cfg['full_policy_or_full_development_automatic_start']
            or cfg['dev_final_Val_Test_or_new_nonKnown_semantics']
            or cfg['model_selection_or_goal_acceptance']):
        raise ValueError('Exact frozen bounded joint Train-only pilot required')
    if sorted(cfg['pilot_policy_relative_ordinals']) != [0, 1]:
        raise ValueError('Fixed two-video pilot scope required')
    return proof


def supervise(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    receipt = out.with_name(out.name + '_supervisor_attempt_1.json')
    if out.exists() or receipt.exists():
        raise ValueError('Preserve every previous joint pilot attempt')
    started, peak, threads, reason = time.monotonic(), 0, 0, None
    if resources_ok(cfg, safety, ROOT, started, 0):
        raise RuntimeError('Pre-spawn resource guard')
    child = subprocess.Popen([
        sys.executable, '-B', '-X', 'pycache_prefix=' + str(out / '_fresh_no_write_cache'),
        str(ROOT / 'scripts/trackocd_cvpr/pilot_joint_rc_policy.py'), '--worker',
        '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy,
    ], cwd=ROOT)
    try:
        while child.poll() is None:
            status_path = Path(f'/proc/{child.pid}/status')
            try:
                status = status_path.read_text()
                peak = max(peak, process_memory(status, Path(f'/proc/{child.pid}/smaps_rollup').read_text())['guarded_process_RSS_bytes'])
                fields = {line.split(':', 1)[0]: line.split(':', 1)[1] for line in status.splitlines() if ':' in line}
                threads = max(threads, int(fields['Threads'].strip()))
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                if child.poll() is not None:
                    break
                try:
                    settled = kernel_terminal_status(status_path.read_text())
                except (FileNotFoundError, ProcessLookupError):
                    settled = False
                if settled:
                    child.wait(timeout=1)
                    break
                if time.monotonic() - started > 2:
                    reason = 'owned_memory_counter_unavailable'
                    child.kill()
                    break
                time.sleep(.2)
                continue
            reason = resources_ok(cfg, safety, ROOT, started, peak)
            if out.exists() and sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > cfg['output_bytes_ceiling']:
                reason = 'owned_output_size'
            if reason:
                if child.poll() is None:
                    child.kill()
                break
            time.sleep(.2)
        code = child.wait(timeout=10)
    except BaseException:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        raise
    atomic_json(receipt, {
        'owned_worker_exit_code': code, 'guard_stop_reason': reason,
        'successful_owned_execution': code == 0 and reason is None,
        'only_new_owned_Popen_worker_managed': True,
        'peak_sampled_owned_worker_RSS_bytes': peak,
        'peak_sampled_worker_threads': threads,
        'wall_seconds_including_imports_network': time.monotonic() - started,
        'training_GPU_hours': 0,
    })
    return 0 if code == 0 and reason is None else 1


def worker(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(out / '_fresh_no_write_cache'):
        raise ValueError('Fresh isolated no-write worker required')
    proof = _source_gate(args, cfg, safety)
    if out.exists():
        raise ValueError('No output overwrite')
    out.mkdir(parents=True)
    started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*')
                      if p.is_file() and not p.is_symlink())

    def guard():
        nonlocal peak
        peak = max(peak, process_memory()['guarded_process_RSS_bytes'])
        reason = resources_ok(cfg, safety, ROOT, started, peak)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes + size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned joint pilot guard: ' + str(reason or 'output_storage'))

    atomic_json(out / 'launch.json', {
        'live_remote_registration_proof': proof,
        'source_sha256': {path: sha256_file(ROOT / path) for path in SOURCES},
        'scope': 'Fixed first two policy-order videos, full selected clocks, two joint R/C optimizer updates, no dev/final evaluation',
    })
    try:
        import torch
        from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
        from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks, support_groups
        from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
        from src.trackocd_cvpr.batched_evidence import reencode_batched_supports
        from src.trackocd_cvpr.joint_rc_policy_training import train_joint_policy_pilot
        from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        def barrier(event, values):
            if event == 'socket.connect':
                raise PermissionError('No network after source/authority registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                path = Path(os.fsdecode(values[0])).resolve()
                if path.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (
                        path.suffix in ('.pt', '.pth') and path.parent != out):
                    raise PermissionError('Qualified caches only; no original pixels/annotations/old weights')
        sys.addaudithook(barrier)

        guard()
        videos, projection, policy_proof = load_completed_policy_videos(ROOT, cfg, guard)
        fit_tracks, fit_proof = load_completed_fit_tracks(ROOT, cfg, guard)
        indices = cfg['pilot_policy_relative_ordinals']
        if max(indices) >= len(videos):
            raise ValueError('Fixed pilot ordinal outside qualified policy order')
        pilot_videos = [videos[index] for index in indices]
        if len({video for video, _ in pilot_videos}) != len(pilot_videos):
            raise ValueError('Pilot video order is not unique')
        known_supports = support_groups(fit_tracks, -1, 5)
        raw_model = RawPreservingEvidence().eval().requires_grad_(False)
        with torch.inference_mode():
            raw, _, _, _ = reencode_batched_supports(raw_model, known_supports,
                                                     torch.ones(5), torch.ones(5))
        known = {index: row.numpy().copy() for index, row in enumerate(raw)}

        def save_checkpoint(evidence, scorer, optimizer, step, episode, episode_step, rng_state):
            path = out / ('private_joint_step' + str(step) + '_checkpoint.pt')
            temporary = out / (path.name + '.partial')
            with temporary.open('xb') as handle:
                torch.save({'evidence_state': evidence.state_dict(), 'scorer_state': scorer.state_dict(),
                            'optimizer_state': optimizer.state_dict(), 'step': step, 'episode': episode,
                            'episode_step': episode_step, 'numpy_rng_state': rng_state}, handle)
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, path)

        def progress(row, curve):
            atomic_json(out / 'progress.json', row)
            atomic_json(out / 'private_loss_curve.json', curve)

        evidence, scorer, optimizer, summary, sealed = train_joint_policy_pilot(
            pilot_videos, projection, known, known_supports, cfg, guard, progress, save_checkpoint)
        if any(row['videos'] != len(pilot_videos) or not row['complete_clock_consumed']
               for row in summary['episodes']):
            raise ValueError('Full declared pilot clock required')
        endpoint = out / 'private_joint_step2_checkpoint.pt'
        if not endpoint.is_file():
            raise ValueError('Fixed joint pilot endpoint missing')
        reopened = torch.load(endpoint, map_location='cpu', weights_only=True)
        evidence_check = RawPreservingEvidence(); scorer_check = type(scorer)(feature_count=14)
        evidence_check.load_state_dict(reopened['evidence_state'], strict=True)
        scorer_check.load_state_dict(reopened['scorer_state'], strict=True)
        if digest(evidence_check) != digest(evidence) or digest(scorer_check) != digest(scorer):
            raise ValueError('Private joint endpoint did not reopen exactly')
        result = {
            'status': 'COMPLETED_JOINT_RC_MEMORY_TRAIN_PILOT_NOT_DEV_PRED_OR_GOAL_ACCEPTANCE',
            'live_remote_registration_proof': proof,
            'policy_scope': {key: value for key, value in policy_proof.items()
                             if key not in ('private_label_bindings',)},
            'offline_fit_prior_scope': fit_proof,
            'pilot_scope': {'policy_relative_ordinals': indices, 'videos': len(pilot_videos),
                            'selection_frozen_before_payload_read': True,
                            'not_development_or_final': True},
            'summary': summary,
            'joint_parameter_counts': {'evidence': sum(p.numel() for p in evidence.parameters()),
                                       'scorer': sum(p.numel() for p in scorer.parameters())},
            'private_endpoint': {'bytes': endpoint.stat().st_size, 'sha256': sha256_file(endpoint),
                                 'content_uploaded': False, 'reopened_exact': True},
            'dev_final_Val_Test_or_new_nonKnown_semantics_used': False,
            'model_selection_or_goal_acceptance': False,
            'resources': {'peak_guarded_RSS_bytes': peak,
                          'wall_seconds_excluding_network': time.monotonic() - started,
                          'training_GPU_hours': 0, 'CUDA_initialized': torch.cuda.is_initialized()},
        }
        atomic_json(out / 'result.json', result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__,
                                           'error': str(error), 'complete': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    parser.add_argument('--git-proxy', default='')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / CONFIG).read_text())
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker:
        worker(args, cfg, safety)
        return 0
    return supervise(args, cfg, safety)


if __name__ == '__main__':
    sys.exit(main())
