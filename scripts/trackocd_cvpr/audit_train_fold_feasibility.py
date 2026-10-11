#!/usr/bin/env python3
"""Registered necessary-only class/video support hypotheses; no model or GT expansion."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.train_fold_feasibility import cyclic_role_feasibility
from src.trackocd_cvpr.train_protocol_inventory import read_registered_known_metadata

SOURCES = ('configs/trackocd_cvpr/cvpr2_train_fold_feasibility.json',
           'scripts/trackocd_cvpr/audit_train_fold_feasibility.py',
           'src/trackocd_cvpr/train_fold_feasibility.py', 'src/trackocd_cvpr/train_protocol_inventory.py',
           'src/trackocd_v2/io.py', 'configs/trackocd_cvpr/autonomous_research.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    parser.add_argument('--git-proxy', default='')
    args = parser.parse_args()
    cfg = json.loads((ROOT / SOURCES[0]).read_text())
    safety = json.loads((ROOT / SOURCES[-1]).read_text())
    git = ['git'] + (['-c', f'http.proxy={args.git_proxy}'] if args.git_proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote = subprocess.check_output(git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"],
        cwd=ROOT, text=True, timeout=35, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()
    if not remote or remote[0] != args.preregistration_commit:
        raise ValueError('Exact current remote registration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{args.preregistration_commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered source changed')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve every attempt')
    started = time.monotonic()

    def guard():
        mem = {key: int(value.split()[0]) * 1024 for key, value in
               (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']:
            raise RuntimeError('RAM reserve')
        if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg['rss_bytes_ceiling']
                or time.monotonic() - started > cfg['wall_seconds_ceiling']
                or shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes']):
            raise RuntimeError('RSS/wall/disk reserve')
        if sum(path.stat().st_size for path in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if path.is_file()) > safety['incremental_storage_hard_bytes']:
            raise RuntimeError('Project storage bound')
        if out.exists() and sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) > cfg['output_bytes_ceiling']:
            raise RuntimeError('Small receipt bound')

    guard()
    out.mkdir(parents=True)
    launch = {'preregistration_commit': args.preregistration_commit,
              'source_sha256': {name: sha256_file(ROOT / name) for name in SOURCES},
              'old_reference_manifest_sha256': cfg['cache_manifest_sha256']}
    atomic_json(out / 'launch.json', launch)
    try:
        rows, scope = read_registered_known_metadata(ROOT / cfg['cache_directory'], cfg['cache_manifest_sha256'])
        if (len(rows), len({row['video_id'] for row in rows}), scope['eligible_old_representation_fit_categories']) != (
                cfg['expected_safe_tracks'], cfg['expected_safe_videos'], cfg['expected_eligible_classes']):
            raise ValueError('Registered permitted universe differs')
        summary = cyclic_role_feasibility(rows, seed=cfg['seed'], folds=cfg['folds'],
            policy_pseudo_classes=cfg['policy_pseudo_classes'],
            minimum_fit_candidate_videos=cfg['minimum_fit_candidate_videos'],
            minimum_eval_candidate_videos=cfg['minimum_eval_candidate_videos'])
        guard()
        result = {'status': 'COMPLETE_FIXED_ROLE_NECESSARY_FEASIBILITY_DIAGNOSTIC_NOT_PROTOCOL_ACCEPTANCE',
                  **launch, 'metadata_scope': scope, 'summary': summary, 'optimization_steps': 0,
                  'model_or_new_annotation_or_feature_execution': False, 'Val_or_Test_access': False,
                  'interpretation': cfg['interpretation'],
                  'resources': {'wall_seconds': time.monotonic() - started,
                                'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                'training_GPU_hours': 0}}
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'fold_necessary_support_deficits': [
            fold['necessary_support_deficits'] for fold in summary['folds']], 'resources': result['resources']}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error),
                                         'complete': False, 'wall_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
