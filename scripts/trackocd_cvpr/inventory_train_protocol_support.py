#!/usr/bin/env python3
"""Registered external Train-metadata feasibility inventory; never trains."""
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
from src.trackocd_cvpr.train_protocol_inventory import read_registered_known_metadata, support_summary

SOURCES = ('configs/trackocd_cvpr/cvpr2_train_protocol_inventory.json',
           'scripts/trackocd_cvpr/inventory_train_protocol_support.py',
           'src/trackocd_cvpr/train_protocol_inventory.py', 'src/trackocd_v2/io.py',
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
        raise ValueError('Preserve attempts; no output overwrite')
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
            raise RuntimeError('Incremental storage bound')
        if out.exists() and sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) > cfg['output_bytes_ceiling']:
            raise RuntimeError('Aggregate receipt size bound')

    guard()
    manifest_path = ROOT / cfg['predicted_feature_manifest']
    if sha256_file(manifest_path) != cfg['predicted_feature_manifest_sha256']:
        raise ValueError('Retained predicted manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if not manifest['complete']:
        raise ValueError('Only completed retained predicted universe')
    videos = {record['video_id'] for record in manifest['records']}
    if len(videos) != cfg['expected_predicted_videos'] or len(videos) != len(manifest['records']):
        raise ValueError('Retained universe mismatch')
    out.mkdir(parents=True)
    launch = {'preregistration_commit': args.preregistration_commit,
              'source_sha256': {name: sha256_file(ROOT / name) for name in SOURCES},
              'old_reference_manifest_sha256': cfg['cache_manifest_sha256'],
              'predicted_feature_manifest_sha256': cfg['predicted_feature_manifest_sha256']}
    atomic_json(out / 'launch.json', launch)
    try:
        rows, metadata = read_registered_known_metadata(ROOT / cfg['cache_directory'], cfg['cache_manifest_sha256'])
        guard()
        if metadata['eligible_old_representation_fit_categories'] != cfg['expected_old_fit_categories']:
            raise ValueError('Known fit pool changed')
        summary = support_summary(rows, videos)
        guard()
        result = {'status': 'COMPLETE_EXTERNAL_TRAIN_REFERENCE_FEASIBILITY_INVENTORY_NOT_NEW_PROTOCOL',
                  **launch, 'metadata_scope': metadata, 'summary': summary,
                  'predicted_cache_video_count': len(videos), 'GT_forward_or_model_execution': False,
                  'Val_or_Test_or_new_GT_annotation_access': False, 'new_asset_download_or_features': False,
                  'old_adapted_weights_eligible_as_unseen_class_baselines': False,
                  'optimization_steps': 0, 'interpretation': cfg['purpose'],
                  'resources': {'wall_seconds': time.monotonic() - started,
                                'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                'training_GPU_hours': 0, 'disk_free_bytes': shutil.disk_usage(ROOT).free}}
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'categories': summary['categories'],
                          'permitted_reference_pool': summary['permitted_reference_pool'],
                          'resources': result['resources']}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error),
                                         'complete': False, 'wall_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
