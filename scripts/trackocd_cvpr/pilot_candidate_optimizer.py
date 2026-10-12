#!/usr/bin/env python3
"""Registered bounded C plumbing on full policy GT input; not a C/M result."""
import argparse
from dataclasses import asdict
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
from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory
from scripts.trackocd_cvpr.qualify_policy_training_input import supervise

CONFIG = 'configs/trackocd_cvpr/candidate_optimizer_pilot_v1.json'
SOURCES = (CONFIG, 'scripts/trackocd_cvpr/pilot_candidate_optimizer.py',
    'scripts/trackocd_cvpr/qualify_policy_training_input.py',
    'src/trackocd_cvpr/candidate_fit_pilot.py', 'src/trackocd_cvpr/candidate_training_loss.py',
    'src/trackocd_cvpr/candidate_score_contract.py', 'src/trackocd_cvpr/atomic_member_control.py',
    'src/trackocd_cvpr/causal_observation_router.py', 'src/trackocd_cvpr/prefix_member_bank.py',
    'src/trackocd_cvpr/policy_training_data.py', 'src/trackocd_cvpr/fit_R_training.py',
    'src/trackocd_cvpr/fit_loss_supervision.py', 'src/trackocd_cvpr/batched_evidence.py',
    'src/trackocd_cvpr/exact_support_reduction.py', 'src/trackocd_cvpr/differentiable_support_bank.py',
    'src/trackocd_cvpr/raw_preserving_evidence.py', 'src/trackocd_cvpr/GT_video_cache.py',
    'src/trackocd_core/features.py', 'src/trackocd_cvpr/scoped_train_census.py',
    'src/trackocd_cvpr/metadata_only.py', 'src/trackocd_cvpr/remote_registration.py', 'src/trackocd_v2/io.py',
    'scripts/trackocd_cvpr/smoke_selected_frozen_encoder.py',
    'scripts/trackocd_cvpr/audit_joint_known_protocol.py', 'scripts/trackocd_cvpr/audit_train_metadata_containment.py',
    'configs/trackocd_cvpr/formal_research_scope_approval_v1.json',
    'configs/trackocd_cvpr/frozen_global_data_GT_control_v1.json',
    'outputs/trackocd_cvpr/cvpr2/fit_GT_video_cache_v1/result.json',
    'outputs/trackocd_cvpr/cvpr2/fit_GT_video_cache_v1_supervisor_attempt_1.json',
    'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v3_resource_retry/result.json',
    'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v3_resource_retry_supervisor_attempt_1.json',
    'outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1/result.json',
    'outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1_supervisor_attempt_1.json',
    'configs/trackocd_cvpr/autonomous_research.json')


def worker(args, cfg, safety):
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(ROOT / cfg['output_directory'] / '_fresh_no_write_cache'):
        raise ValueError('Fresh no-write worker required')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for path in SOURCES:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('C source/cache/input proof not actually preregistered')
    authority = json.loads((ROOT / cfg['approval']).read_text())
    if (authority['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or cfg['status'] != 'REGISTERED_TEN_C_STEPS_COMPLETE_POLICY_ROLLOUT_NOT_SCIENTIFIC_ACCEPTANCE'
            or cfg['optimizer_steps'] != 10 or cfg['seed'] != 1027 or cfg['K_per_family'] != 8
            or cfg['execution_device'] != 'CPU' or cfg['full_scientific_fit_automatic_start'] is not False):
        raise ValueError('Exact legal first bounded C optimizer pilot')
    qualification = ROOT / 'outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1'
    terminal = json.loads(qualification.with_name(qualification.name + '_supervisor_attempt_1.json').read_text())
    if (sha256_file(qualification / 'result.json') != cfg['policy_input_qualification_sha256']
            or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None):
        raise ValueError('Actual successful full policy qualification required')
    out = ROOT / cfg['output_directory']
    if out.exists(): raise ValueError('No prior output overwrite')
    out.mkdir(parents=True); started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, process_memory()['guarded_process_RSS_bytes'])
        reason = resources_ok(cfg, safety, ROOT, started, peak)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes + size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned C pilot guard:' + str(reason))
    atomic_json(out / 'launch.json', {'live_remote_registration_proof': proof,
        'source_sha256': {p: sha256_file(ROOT / p) for p in SOURCES}, 'scope': 'Ten real C steps, full policy GT rollout, no scientific acceptance'})
    try:
        import torch
        from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
        from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks, support_groups
        from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
        from src.trackocd_cvpr.batched_evidence import reencode_batched_supports
        from src.trackocd_cvpr.fit_loss_supervision import project_fit_loss_targets
        from src.trackocd_cvpr.candidate_fit_pilot import fit_candidate_pilot
        from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
        from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
        torch.set_num_threads(1)
        def barrier(event, values):
            if event == 'socket.connect': raise PermissionError('No network after source registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                path = Path(os.fsdecode(values[0])).resolve()
                if path.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (
                        path.suffix in ('.pt', '.pth') and path.parent != out):
                    raise PermissionError('Qualified fit/policy cache only, no original data or old weights')
        sys.addaudithook(barrier)
        guard(); videos, targets, policy_proof = load_completed_policy_videos(ROOT, cfg, guard)
        fit_tracks, fit_proof = load_completed_fit_tracks(ROOT, cfg, guard)
        bindings, _, _ = project_fit_loss_targets(ROOT / cfg['private_assignment'], ROOT / cfg['private_Known_support'])
        if bindings != targets.fit_bindings: raise ValueError('Exact offline Known prior / policy loss binding order')
        torch.manual_seed(cfg['seed']); raw_model = RawPreservingEvidence().eval().requires_grad_(False)
        with torch.inference_mode():
            raw, _, _, _ = reencode_batched_supports(raw_model, support_groups(fit_tracks, -1, 5), torch.ones(5), torch.ones(5))
        known = {i: row.numpy().copy() for i, row in enumerate(raw)}
        model, optimizer, summary, sealed, history = fit_candidate_pilot(videos, targets, known, cfg, guard,
            lambda row: atomic_json(out / 'progress.json', row))
        if summary['complete_rollout']['physical_observations'] != 1268 or summary['complete_rollout']['metadata_frames'] != 374:
            raise ValueError('Complete physical policy rollout, no loss-only subset')
        endpoint = out / 'private_step10_checkpoint.pt'; temporary = out / 'private_step10_checkpoint.partial'
        with temporary.open('xb') as handle:
            torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                        'step': 10, 'seed': cfg['seed'], 'torch_rng_state': torch.get_rng_state()}, handle)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, endpoint)
        after = digest(model); verifier = CandidateScoreMLP()
        verifier.load_state_dict(torch.load(endpoint, map_location='cpu', weights_only=True)['model_state'], strict=True)
        if digest(verifier) != after or torch.cuda.is_initialized(): raise ValueError('Exact real CPU C endpoint reopen')
        video_for_frames = [video for video, arrays in videos for _ in arrays['clock_frame_index']]
        if len(video_for_frames) != len(sealed): raise ValueError('Every sealed frame binds to its actual video')
        history_path = out / 'private_prediction_history.json'
        atomic_json(history_path, {'anonymous_commit_labels': [asdict(r) for r in history],
            'actual_predictions': [{'video_id': video, 'frame_index': r.frame_index, 'routing_groups': r.routing_groups,
                'events': [{'action': e.action, 'category_id': e.category_id, 'token': e.token,
                            'ambiguity_reason': e.ambiguity_reason} for e in r.events]} for video, r in zip(video_for_frames, sealed)]})
        guard()
        result = {'status': 'COMPLETED_TEN_C_OPTIMIZER_STEPS_FULL_POLICY_ROLLOUT_NOT_C_M_SCIENTIFIC_ACCEPTANCE',
            'live_remote_registration_proof': proof, 'seed': cfg['seed'], 'summary': summary,
            'policy_scope': policy_proof, 'offline_fit_prior_scope': fit_proof,
            'offline_known_prior_and_policy_loss_binding_order_exact': True,
            'endpoint': {'bytes': endpoint.stat().st_size, 'sha256': sha256_file(endpoint), 'reopen_exact': True},
            'private_prediction_history': {'bytes': history_path.stat().st_size, 'sha256': sha256_file(history_path)},
            'model_parameter_sha256_after': after, 'dev_final_Val_Test_or_new_nonKnown_labels_used': False,
            'all3seeds_or_distinct_R_C_M_or_qualified_Pred_or_Goal_accepted': False,
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
    return supervise(args, cfg, safety, worker_script='scripts/trackocd_cvpr/pilot_candidate_optimizer.py')


if __name__ == '__main__': sys.exit(main())
