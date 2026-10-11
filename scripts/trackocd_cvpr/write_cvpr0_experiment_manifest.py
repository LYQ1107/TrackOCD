#!/usr/bin/env python3
"""Aggregate actual stage receipts only; never synthesize missing metrics."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
DIRECTORIES = ('historical_audit_integer_normalized', 'checkpoint_replay_v1',
    'memory_controls_full_denominator_v2', 'train_input_preflight_v1',
    'train_physical_smoke_v1', 'retained_logit_forensics_v1', 'state_trajectories_v1',
    'cross_video_state_forensics_v1', 'train_physical_diagnostic_v1')


def verified_sources(root, commit, source_sha256):
    if len(commit) != 40:
        raise ValueError('Exact preregistration SHA, not guessed abbreviation')
    for name, digest in source_sha256.items():
        content = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('Actual receipt differs from registered source bytes')
    return True


def build(root):
    phase = root / 'outputs/trackocd_cvpr/cvpr0'
    stages = []
    for name in DIRECTORIES:
        directory = phase / name
        result_path, registration = directory / 'result.json', directory / 'registration.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            # Physical worker receipts do not repeat launch proof; use its actual immutable registration.
            explicit_commit = 'preregistration_commit' in result
            if name == 'train_input_preflight_v1':
                # Do not invent a launch receipt or alter the measured old result.
                proof = {**result, 'preregistration_commit': '18e6c65104d0005fb368a0d9eb5b29656c561fb9'}
                config_bytes = subprocess.check_output(['git', 'show',
                    f"{proof['preregistration_commit']}:configs/trackocd_cvpr/cvpr0_train_input_preflight.json"], cwd=root)
                if hashlib.sha256(config_bytes).hexdigest() != result['config_sha256']:
                    raise ValueError('Logged preflight registration configuration binding differs')
                binding = 'POSTHOC_BYTE_BINDING_TO_LOGGED_PREEXECUTION_COMMIT; original receipt has no commit field'
            elif not explicit_commit:
                registration = directory / ('registration.json' if registration.exists() else 'launch.json')
                proof = json.loads(registration.read_text())
                binding = 'IMMUTABLE_ACTUAL_LAUNCH_REGISTRATION'
            else:
                proof = result
                binding = 'ACTUAL_RESULT_PREREGISTRATION_RECEIPT'
            commit, sources = proof['preregistration_commit'], proof['source_sha256']
            verified_sources(root, commit, sources)
            stages.append({'id': name, 'state': 'COMPLETE_SUBSTEP_NOT_GOAL_COMPLETE', 'actual_status': result['status'],
                'result_path': str(result_path.relative_to(root)), 'result_bytes': result_path.stat().st_size,
                'result_sha256': sha256_file(result_path), 'preregistration_commit': commit,
                'registration_binding': binding, 'original_result_contains_commit': explicit_commit,
                'registered_source_sha256': sources, 'source_receipt_matches_registered_git_bytes': True,
                'resources': result.get('resources'),
                'public_artifacts': [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
                                     for p in sorted(directory.glob('*.csv'))],
                'method_success_inferred_from_diagnostic': False})
        elif registration.exists():
            proof = json.loads(registration.read_text())
            verified_sources(root, proof['preregistration_commit'], proof['source_sha256'])
            progress = json.loads((directory / 'progress.json').read_text()) if (directory / 'progress.json').exists() else {}
            stages.append({'id': name, 'state': 'REGISTERED_RUNNING_NO_COMPLETED_METRIC',
                'preregistration_commit': proof['preregistration_commit'],
                'registered_source_sha256': proof['source_sha256'], 'progress_snapshot': progress,
                'completed_result_present': False, 'method_success_inferred_from_diagnostic': False})
        else:
            stages.append({'id': name, 'state': 'NOT_EXECUTED_NO_METRICS'})
    return {'schema_version': 'trackocd.cvpr0.experiment-manifest.v1', 'status': 'IN_PROGRESS',
        'manifest_generator_source_sha256': sha256_file(root / 'scripts/trackocd_cvpr/write_cvpr0_experiment_manifest.py'),
        'historical_parent': '7546527e5f9540d351785ade34c8e3292a74441d',
        'stages': stages, 'training_GPU_hours_used': 0,
        'no_unique_causal_root_or_new_method_success_claim': True,
        'preserved_failed_attempts': [
            {'path': 'outputs/trackocd_cvpr/cvpr0/historical_audit', 'reason': 'NumPy integer JSON serialization failure', 'not_deleted': True},
            {'path': 'outputs/trackocd_cvpr/cvpr0/memory_controls_v1', 'reason': 'Known denominator registry mismatch correctly refused', 'not_deleted': True}],
        'pending': ['matched semantic predicted-vs-truncated-GT comparison',
                    'fixed-history premature-commit counterfactual', 'ID-independent dense online protocol',
                    '12-source actual-code review', 'new independent Train protocol and at least3mechanisms', 'Goals A/B/C']}


if __name__ == '__main__':
    result = build(ROOT)
    atomic_json(ROOT / 'outputs/trackocd_cvpr/cvpr0/experiment_manifest.json', result)
    print(json.dumps({'status': result['status'], 'stages': len(result['stages']),
                      'completed_substeps': sum(x['state'].startswith('COMPLETE') for x in result['stages'])}))
