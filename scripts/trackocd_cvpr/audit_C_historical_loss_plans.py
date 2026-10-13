#!/usr/bin/env python3
"""Owned read-only ALL9 C loss-plan reconstruction; no score model or fitting.

Existing C endpoints are byte-hashed by the cohort gate, never deserialized.
Only hash-qualified histories and lawful fit/policy caches/labels are decoded.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import shadow_M_raw_cache as runtime
from scripts.trackocd_cvpr import pilot_candidate_optimizer as training_source
from scripts.trackocd_cvpr import evaluate_C_development as C_cohort
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration

SCRIPT = 'scripts/trackocd_cvpr/audit_C_historical_loss_plans.py'
CONFIG = 'configs/trackocd_cvpr/C_historical_loss_plan_audit_v1.json'
BANK = 'src/trackocd_cvpr/prefix_member_bank.py'
SOURCES = tuple(dict.fromkeys((*training_source.SOURCES, *C_cohort.BASE_SOURCES,
    SCRIPT, CONFIG, 'src/trackocd_cvpr/C_historical_loss_plan_audit.py',
    'scripts/trackocd_cvpr/shadow_M_raw_cache.py', *C_cohort.cohort_sources())))


def bank_without_copy_method(source):
    tree = ast.parse(source)
    bank = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PrefixMemberBank')
    bank.body = [node for node in bank.body if not isinstance(node, ast.FunctionDef) or node.name != '__deepcopy__']
    return ast.dump(tree, include_attributes=False)


def checked_configuration(cfg):
    if (cfg['status'] != 'REGISTERED_ALL9_READONLY_C_RECORDED_ACTION_LOSS_PLAN_RECONSTRUCTION'
            or cfg['data_config'] != 'configs/trackocd_cvpr/candidate_optimizer_pilot_v1.json'
            or cfg['window'] != 'configs/trackocd_cvpr/C_GT_scientific_window_v1.json'
            or cfg['copy_only_bank_reference_commit'] != 'e9c894ed0b3ef70b3fc1bc9651893de7a38b3030'
            or cfg['old_bank_sha256'] != 'f2d40b4062ac3e13950654ad467a6ebb7f8dbbf0aeedecc14f2eba7240f007d3'
            or cfg['current_bank_sha256'] != 'a454f7192bacd5d4762d468b46a08afd6b2f94d27ea755e5b1ccd56b81a03e71'
            or cfg['variants'] != list(C_cohort.VARIANTS) or cfg['seeds'] != list(C_cohort.SEEDS)
            or (cfg['episodes_each'], cfg['historical_updates_each'], cfg['new_optimizer_steps']) != (4, 200, 0)
            or cfg['maximum_private_history_bytes_each'] != 2097152
            or (cfg['execution_device'], cfg['CPU_compute_workers'], cfg['rss_bytes_ceiling'],
                cfg['wall_seconds_ceiling'], cfg['output_bytes_ceiling']) != ('CPU', 1, 3221225472, 3000, 3145728)
            or cfg['output_directory'] != 'outputs/trackocd_cvpr/cvpr3/C_historical_loss_plan_audit_v1'
            or cfg['model_checkpoint_deserialization_new_labels_pixels_dev_final_Val_Test_allowed']
            or cfg['historical_logits_gradient_magnitudes_exact_numeric_row_trace_or_collapse_cause_recovered']
            or cfg['private_history_or_raw_GT_upload'] or cfg['scientific_Goal_acceptance']):
        raise ValueError('Exact bounded full ALL9 read-only reconstruction scope')


def source_gate(args, cfg, safety):
    checked_configuration(cfg)
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for path in SOURCES:
        if (ROOT / path).is_symlink() or subprocess.check_output(
                ['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Actual full C source/cohort/protocol not registered')
    data = json.loads((ROOT / cfg['data_config']).read_text())
    authority = json.loads((ROOT / data['approval']).read_text())
    if authority['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN':
        raise ValueError('Existing frozen fit/policy Known scope authority required')
    old = subprocess.check_output(['git', 'show', cfg['copy_only_bank_reference_commit'] + ':' + BANK], cwd=ROOT)
    if (hashlib.sha256(old).hexdigest() != cfg['old_bank_sha256']
            or sha256_file(ROOT / BANK) != cfg['current_bank_sha256']
            or bank_without_copy_method(old) != bank_without_copy_method((ROOT / BANK).read_bytes())):
        raise ValueError('Exact declared copy-only bank source, no changed numeric/provenance logic')
    return proof, SOURCES, data


def worker(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(out / '_fresh_no_write_cache'):
        raise ValueError('Fresh own no-write worker')
    proof, sources, data = source_gate(args, cfg, safety)
    cohort = C_cohort.verified_cohort(ROOT)
    for (variant, seed), (checkpoint, receipt) in cohort.items():
        path = checkpoint.parent / 'private_prediction_history.json'
        if (path.is_symlink() or path.parent.is_symlink()
                or path.stat().st_size != receipt['private_prediction_history']['bytes']
                or path.stat().st_size > cfg['maximum_private_history_bytes_each']
                or sha256_file(path) != receipt['private_prediction_history']['sha256']):
            raise ValueError('All immutable bounded actual C histories before any decode')
        launch = json.loads((checkpoint.parent / 'launch.json').read_text())
        for name in ('src/trackocd_cvpr/atomic_member_control.py', 'src/trackocd_cvpr/causal_observation_router.py',
                     'src/trackocd_cvpr/batched_evidence.py', 'src/trackocd_cvpr/raw_preserving_evidence.py',
                     'src/trackocd_core/features.py'):
            if launch['source_sha256'][name] != sha256_file(ROOT / name):
                raise ValueError('Exact original numerical replay/prior implementation')
        if launch['source_sha256'][BANK] != cfg['old_bank_sha256']:
            raise ValueError('Exact recorded original bank, no blanket copy exception')
    if out.exists(): raise ValueError('Preserve previous audit namespace')
    out.mkdir(parents=True); started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, runtime.process_memory()['guarded_process_RSS_bytes'])
        reason = runtime.resources_ok(cfg, safety, ROOT, started, peak)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes + size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned C historical audit guard:' + str(reason))
    atomic_json(out / 'launch.json', {'live_remote_registration_proof': proof,
        'source_sha256': {p: sha256_file(ROOT / p) for p in sources},
        'scope': 'Actual recorded policy actions/plans; no logits/gradients recovered'})
    try:
        import torch
        from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
        from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks, support_groups
        from src.trackocd_cvpr.fit_loss_supervision import project_fit_loss_targets
        from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
        from src.trackocd_cvpr.batched_evidence import reencode_batched_supports
        from src.trackocd_cvpr.C_historical_loss_plan_audit import audit_recorded_C_trial
        torch.set_num_threads(1)
        def barrier(event, values):
            if event == 'socket.connect': raise PermissionError('No network after registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                path = Path(os.fsdecode(values[0])).resolve()
                if path.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or path.suffix in ('.pt', '.pth'):
                    raise PermissionError('Qualified fit/policy caches/history only, no original data/weights')
        sys.addaudithook(barrier)
        guard(); videos, targets, policy_scope = load_completed_policy_videos(ROOT, data, guard)
        fit, fit_scope = load_completed_fit_tracks(ROOT, data, guard)
        bindings, _, _ = project_fit_loss_targets(ROOT / data['private_assignment'], ROOT / data['private_Known_support'])
        if bindings != targets.fit_bindings: raise ValueError('Exact original fit/policy Known binding order')
        with torch.inference_mode():
            raw, _, _, _ = reencode_batched_supports(RawPreservingEvidence().eval().requires_grad_(False),
                support_groups(fit, -1, 5), torch.ones(5), torch.ones(5))
        known = {i: row.numpy().copy() for i, row in enumerate(raw)}
        external = {(v, str(t)): label for (v, t), label in targets.targets.items()}
        window = json.loads((ROOT / cfg['window']).read_text()); table = {}
        for (variant, seed), (checkpoint, receipt) in cohort.items():
            guard(); path = checkpoint.parent / 'private_prediction_history.json'
            if sha256_file(path) != receipt['private_prediction_history']['sha256']:
                raise ValueError('Actual history changed after initial qualification')
            private = json.loads(path.read_text())
            measured = audit_recorded_C_trial(videos, external, known, window, variant, seed,
                private, receipt['summary'], guard)
            table[variant + '_s' + str(seed)] = measured; del private
            atomic_json(out / 'progress.json', {'completed_trials': len(table), 'new_optimizer_steps': 0})
        guard()
        if len(table) != 9 or torch.cuda.is_initialized(): raise ValueError('Complete ALL9 CPU audit only')
        result = {'status': 'ACTUAL_ALL9_C_HISTORY_LOSS_PLAN_RECONSTRUCTION_NOT_LOGITS_GRADIENTS_CAUSE_OR_GOAL',
            'live_remote_registration_proof': proof, 'policy_scope': policy_scope, 'offline_fit_scope': fit_scope,
            'trials': table, 'new_optimizer_steps': 0,
            'new_inference_collapse_cause_development_Pred_or_Goal_accepted': False,
            'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_registration': time.monotonic() - started,
                'CUDA_initialized': False, 'training_GPU_hours': 0}}
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'completed_trials': 9}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False}); raise


def main():
    runtime.SCRIPT, runtime.CONFIG = SCRIPT, CONFIG
    runtime.source_gate, runtime.worker = source_gate, worker
    return runtime.main()  # original guarded owned persistent launch, never external PIDs


if __name__ == '__main__': sys.exit(main())
