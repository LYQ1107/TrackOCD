#!/usr/bin/env python3
"""Full9-endpoint gated causal GT dev evaluation; no fitting or winner selection."""
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
from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory
from scripts.trackocd_cvpr.qualify_policy_training_input import supervise

CONFIG = 'configs/trackocd_cvpr/C_GT_causal_development_v1.json'
VARIANTS = ('Top1_conditioned', 'Top8_conditioned', 'Top8_pool_context')
SEEDS = (1027, 1028, 1029)
MODELS = ('Raw_Top1', 'Raw_Top8', *VARIANTS)
BASE_SOURCES = (CONFIG, 'scripts/trackocd_cvpr/evaluate_C_development.py',
    'src/trackocd_cvpr/C_development_rollout.py', 'src/trackocd_cvpr/causal_category_audit.py',
    'src/trackocd_cvpr/R_development_evaluation.py', 'src/trackocd_cvpr/R_development_metrics.py',
    'src/trackocd_cvpr/fit_R_training.py', 'src/trackocd_cvpr/fit_loss_supervision.py',
    'src/trackocd_cvpr/policy_training_data.py', 'src/trackocd_cvpr/candidate_scientific_training.py',
    'src/trackocd_cvpr/candidate_training_loss.py', 'src/trackocd_cvpr/candidate_score_contract.py',
    'src/trackocd_cvpr/atomic_member_control.py', 'src/trackocd_cvpr/causal_observation_router.py',
    'src/trackocd_cvpr/prefix_member_bank.py', 'src/trackocd_cvpr/GT_video_cache.py',
    'src/trackocd_cvpr/batched_evidence.py', 'src/trackocd_cvpr/raw_preserving_evidence.py',
    'src/trackocd_cvpr/differentiable_support_bank.py', 'src/trackocd_cvpr/exact_support_reduction.py',
    'src/trackocd_core/features.py', 'src/trackocd_cvpr/scoped_train_census.py', 'src/trackocd_cvpr/metadata_only.py',
    'src/trackocd_cvpr/remote_registration.py', 'src/trackocd_v2/io.py',
    'scripts/trackocd_cvpr/smoke_selected_frozen_encoder.py',
    'scripts/trackocd_cvpr/audit_joint_known_protocol.py', 'scripts/trackocd_cvpr/audit_train_metadata_containment.py',
    'scripts/trackocd_cvpr/qualify_policy_training_input.py',
    'configs/trackocd_cvpr/C_GT_scientific_window_v1.json',
    'configs/trackocd_cvpr/formal_research_scope_approval_v1.json', 'configs/trackocd_cvpr/frozen_global_data_GT_control_v1.json',
    'outputs/trackocd_cvpr/cvpr2/fit_GT_video_cache_v1/result.json',
    'outputs/trackocd_cvpr/cvpr2/fit_GT_video_cache_v1_supervisor_attempt_1.json',
    'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v3_resource_retry/result.json',
    'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v3_resource_retry_supervisor_attempt_1.json')


def cohort_sources():
    return tuple(path for variant in VARIANTS for seed in SEEDS for path in (
        'outputs/trackocd_cvpr/cvpr3/C_GT_window_v1_' + variant + '_s' + str(seed) + '/launch.json',
        'outputs/trackocd_cvpr/cvpr3/C_GT_window_v1_' + variant + '_s' + str(seed) + '/result.json',
        'outputs/trackocd_cvpr/cvpr3/C_GT_window_v1_' + variant + '_s' + str(seed) + '_supervisor_attempt_1.json'))


def verified_cohort(root):
    results = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            out = root / ('outputs/trackocd_cvpr/cvpr3/C_GT_window_v1_' + variant + '_s' + str(seed))
            r = json.loads((out / 'result.json').read_text())
            t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
            if (r['variant'] != variant or r['seed'] != seed or r['summary']['optimizer_steps'] != 200
                    or r['status'] != 'COMPLETED_FIXED_C_TRIAL_NOT_DEV_M_PRED_OR_GOAL_ACCEPTANCE'
                    or r['summary']['trainable_parameters'] != 708 or r['summary']['GT_repair_or_deleted_wrong_history']
                    or r['checkpoint_selection'] != 'Fixed200 endpoint' or not r['endpoint']['reopen_exact']
                    or r['old_pilot_or_R_or_policy_warmstart'] or r['dev_final_Val_Test_or_new_nonKnown_labels_used']
                    or not t['successful_owned_execution'] or t['owned_worker_exit_code'] != 0 or t['guard_stop_reason'] is not None):
                raise ValueError('ALL9 actual fresh fixed endpoints/owned terminal receipts required')
            launch = json.loads((out / 'launch.json').read_text())
            for path in ('configs/trackocd_cvpr/C_GT_scientific_window_v1.json',
                         'src/trackocd_cvpr/candidate_scientific_training.py',
                         'src/trackocd_cvpr/candidate_training_loss.py', 'src/trackocd_cvpr/candidate_score_contract.py'):
                if launch['source_sha256'][path] != sha256_file(root / path):
                    raise ValueError('Same actual registered C window/kernel/loss, no hidden variant changes')
            checkpoint = out / 'private_step200_checkpoint.pt'
            if checkpoint.is_symlink() or checkpoint.stat().st_size != r['endpoint']['bytes'] or sha256_file(checkpoint) != r['endpoint']['sha256']:
                raise ValueError('Actual private endpoint fingerprint before any dev labels')
            results[(variant, seed)] = (checkpoint, r)
    return results


def worker(args, cfg, safety):
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(ROOT / cfg['output_directory'] / '_fresh_no_write_cache'):
        raise ValueError('Fresh no-write evaluation worker')
    sources = (*BASE_SOURCES, *cohort_sources(), 'configs/trackocd_cvpr/autonomous_research.json')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for path in sources:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Evaluation source/complete cohort/cache proof not actually registered')
    authority = json.loads((ROOT / cfg['approval']).read_text())
    if (authority['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or cfg['status'] != 'REGISTERED_READONLY_CAUSAL_DEV_ALL9_TRAINED_ENDPOINTS_REQUIRED'
            or cfg['optimizer_steps'] != 0 or cfg['select_model_or_checkpoint'] is not False):
        raise ValueError('Only registered read-only development, never model selection')
    cohort = verified_cohort(ROOT)  # AFTER source/authority, BEFORE any private GT projection/feature load.
    out = ROOT / cfg['output_directory']
    if out.exists(): raise ValueError('Preserve all earlier evaluation attempts')
    out.mkdir(parents=True); started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, process_memory()['guarded_process_RSS_bytes'])
        reason = resources_ok(cfg, safety, ROOT, started, peak)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes + size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned causal evaluator guard:' + str(reason))
    atomic_json(out / 'launch.json', {'live_remote_registration_proof': proof,
        'source_sha256': {p: sha256_file(ROOT / p) for p in sources}, 'scope': 'Causal GTdev/all9 completed fits/no selection'})
    try:
        import numpy as np
        import torch
        from src.trackocd_cvpr.C_development_rollout import dev_videos_and_external_labels, readonly_C_stream
        from src.trackocd_cvpr.R_development_evaluation import load_completed_fit_tracks, known_prototypes
        from src.trackocd_cvpr.fit_loss_supervision import project_fit_loss_targets
        from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
        from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
        from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
        torch.set_num_threads(1)
        allowed_checkpoints = {p.resolve() for p, _ in cohort.values()}
        def barrier(event, values):
            if event == 'socket.connect': raise PermissionError('No network after source registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                p = Path(os.fsdecode(values[0])).resolve()
                if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (p.suffix in ('.pt', '.pth') and p not in allowed_checkpoints):
                    raise PermissionError('Qualified cache/own C endpoints only, no original data or encoder weights')
        sys.addaudithook(barrier)
        guard(); videos, labels, scope = dev_videos_and_external_labels(ROOT, cfg, guard)
        fit, fit_scope = load_completed_fit_tracks(ROOT, cfg, guard)
        bindings, _, _ = project_fit_loss_targets(ROOT / cfg['private_assignment'], ROOT / cfg['private_Known_support'])
        raw_model = RawPreservingEvidence().eval().requires_grad_(False)
        raw, _ = known_prototypes(raw_model, fit); known = {i: row for i, row in enumerate(raw)}
        known_bindings = {slot: category for category, slot in bindings.items()}
        model = None
        if args.model in VARIANTS:
            checkpoint, receipt = cohort[(args.model, args.seed)]
            model = CandidateScoreMLP()
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state'], strict=True)
            model.eval().requires_grad_(False)
            if digest(model) != receipt['model_parameter_sha256_after']: raise ValueError('Exact completed scorer')
        table = {}; first_private = None
        for order in cfg['order_seeds']:
            measured, private = readonly_C_stream(videos, labels, known, known_bindings, cfg, args.model, order, guard, model=model)
            table[str(order)] = measured
            if first_private is None: first_private = private
            # Aggregate only on disk; no raw GT/feature/prediction row upload.
            atomic_json(out / 'progress.json', {'completed_order': order, 'physical_tracks': 472, 'no_optimizer': True})
        renamed, renamed_private = readonly_C_stream(videos, labels, known, known_bindings, cfg, args.model,
            cfg['order_seeds'][0], guard, model=model, rename_and_reverse=True)
        if renamed != table[str(cfg['order_seeds'][0])] or [r.events for _, r in renamed_private] != [r.events for _, r in first_private]:
            raise ValueError('Actual full stream IDs/delivery affect numeric policy or outcome audit')
        first_ordinal = int(np.random.default_rng(cfg['order_seeds'][0]).permutation(len(videos))[0])
        v, arrays = videos[first_ordinal]; split = int(arrays['frame_index'].min())
        future = arrays['frame_index'] > split; future_count = int(future.sum())
        perturbation = {'future_observations_mutated': future_count, 'verified_same_prior_events': False}
        if future_count:
            changed = {k: a.copy() for k, a in arrays.items()}; changed['visual'][future] = 0; changed['visual'][future, 0] = 1
            actual_changes = int(np.any(changed['visual'][future] != arrays['visual'][future], axis=1).sum())
            if actual_changes == 0: raise ValueError('Future mutation must change actual numeric observations')
            perturbation['future_observations_mutated'] = actual_changes
            altered = list(videos); altered[first_ordinal] = (v, changed)
            _, altered_private = readonly_C_stream(altered, labels, known, known_bindings, cfg, args.model,
                cfg['order_seeds'][0], guard, model=model)
            left = [r.events for video, r in first_private if video == v and r.frame_index <= split]
            right = [r.events for video, r in altered_private if video == v and r.frame_index <= split]
            if left != right or not any(left): raise ValueError('Nonvacuous actual future-prefix invariance failed')
            perturbation['verified_same_prior_events'] = True
        guard()
        result = {'status': 'MEASURED_CAUSAL_GT_C_DEVELOPMENT_NOT_M_PRED_FULL_TRUTH_OR_GOAL_ACCEPTANCE',
            'live_remote_registration_proof': proof, 'model': args.model, 'training_seed': args.seed if model is not None else None,
            'development_scope': scope, 'offline_fit_prior_scope': fit_scope, 'orders': table,
            'actual_full_ID_rename_reverse_stream_equal': True, 'actual_future_probe': perturbation,
            'optimizer_steps': 0, 'model_or_checkpoint_selection': False,
            'final_Val_Test_or_new_nonKnown_semantic_values_read': False,
            'full_semantic_truth_Hscore_CT_M_Pred_or_Goal_accepted': False,
            'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_network': time.monotonic() - started,
                          'training_GPU_hours': 0, 'CUDA_initialized': torch.cuda.is_initialized()}}
        if result['resources']['CUDA_initialized']: raise ValueError('CPU-only evaluation')
        atomic_json(out / 'result.json', result); print(json.dumps(result), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False}); raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True); parser.add_argument('--git-proxy', default='')
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args(); cfg = json.loads((ROOT / CONFIG).read_text())
    if args.model.startswith('Raw_') and args.seed != 1027: raise ValueError('Raw not fake independent trained seeds')
    cfg['output_directory'] = cfg['output_prefix'] + '_' + args.model + ('_s' + str(args.seed) if args.model in VARIANTS else '')
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker: worker(args, cfg, safety); return 0
    return supervise(args, cfg, safety, worker_script='scripts/trackocd_cvpr/evaluate_C_development.py')


if __name__ == '__main__': sys.exit(main())
