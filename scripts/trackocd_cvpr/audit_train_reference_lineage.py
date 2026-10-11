#!/usr/bin/env python3
"""Metadata/hash-only old reference lineage; no original command re-execution."""
import argparse
import hashlib
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
from src.trackocd_cvpr.metadata_only import train_plan_source_metadata

SOURCES = ('configs/trackocd_cvpr/cvpr2_train_reference_lineage.json',
           'scripts/trackocd_cvpr/audit_train_reference_lineage.py',
           'src/trackocd_cvpr/metadata_only.py', 'src/trackocd_v2/io.py',
           'configs/trackocd_cvpr/autonomous_research.json')


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
            raise RuntimeError('Project incremental bound')
        if out.exists() and sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) > cfg['output_bytes_ceiling']:
            raise RuntimeError('Small receipt bound')

    guard()
    manifest_path, plan_path = ROOT / cfg['cache_manifest'], ROOT / cfg['private_plan']
    if sha256_file(manifest_path) != cfg['cache_manifest_sha256'] or sha256_file(plan_path) != cfg['private_plan_sha256']:
        raise ValueError('Pinned reference manifest/plan differs')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['selection_plan_sha256'] != cfg['private_plan_sha256']
            or manifest['preregistration_commit'] != cfg['historical_registration_commit']):
        raise ValueError('Manifest does not bind registered original plan')
    out.mkdir(parents=True)
    launch = {'preregistration_commit': args.preregistration_commit,
              'source_sha256': {name: sha256_file(ROOT / name) for name in SOURCES},
              'cache_manifest_sha256': cfg['cache_manifest_sha256'], 'private_plan_sha256': cfg['private_plan_sha256']}
    atomic_json(out / 'launch.json', launch)
    try:
        plan = train_plan_source_metadata(plan_path)
        if plan['annotation_sha256'] != cfg['train_annotation_sha256'] or set(plan['sources']) != set(cfg['expected_plan_source_paths']):
            raise ValueError('Unexpected plan source metadata')
        checks = []
        for name in cfg['expected_plan_source_paths'] + cfg['additional_historical_source_paths']:
            guard()
            original = subprocess.check_output(['git', 'show', f"{cfg['historical_registration_commit']}:{name}"], cwd=ROOT)
            digest = hashlib.sha256(original).hexdigest()
            if sha256_file(ROOT / name) != digest:
                raise ValueError('Current helper/source not historical Git byte equivalent')
            captured = name in plan['sources']
            if captured and plan['sources'][name] != digest:
                raise ValueError('Actual recorded plan source differs from registered Git')
            checks.append({'path': name, 'sha256': digest, 'actual_original_plan_source_hash_recorded': captured,
                           'current_and_historical_git_bytes_equal': True,
                           'original_runtime_byte_identity_proven_from_recorded_plan': captured})
        guard()
        result = {'status': 'COMPLETE_NARROW_PLAN_SOURCE_LINEAGE_NOT_INDEPENDENT_GT_CENSUS',
                  **launch, 'historical_registration_commit': cfg['historical_registration_commit'],
                  'source_checks': checks, 'private_plan_bytes': plan_path.stat().st_size,
                  'private_plan_rows_or_summary_JSON_decoded': False, 'role_ID_arrays_JSON_decoded': False,
                  'new_or_protected_GT_labels_decoded': False, 'annotation_file_opened_in_this_audit': False,
                  'Val_or_Test_access': False, 'historical_commands_or_workers_executed': False,
                  'conditional_inference_from_inspected_split_source': 'All old representation-fit category tracks retained in every assigned partition; observations capped after track construction. Not an independent current GT census.',
                  'all_transitive_historical_runtime_dependencies_proven': False,
                  'fresh_protocol_or_complete_category_absence_certified': False, 'optimization_steps': 0,
                  'interpretation': cfg['interpretation'],
                  'resources': {'wall_seconds': time.monotonic() - started,
                                'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                'training_GPU_hours': 0}}
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'recorded_source_checks': sum(row['actual_original_plan_source_hash_recorded'] for row in checks),
                          'additional_current_to_historical_only_checks': sum(not row['actual_original_plan_source_hash_recorded'] for row in checks),
                          'resources': result['resources']}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error),
                                         'complete': False, 'wall_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
