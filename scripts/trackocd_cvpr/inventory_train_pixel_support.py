#!/usr/bin/env python3
"""Preregistered stat-only coverage; no new assets, GPU or annotation arrays."""
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
from src.trackocd_cvpr.metadata_only import image_video_metadata
from src.trackocd_cvpr.train_pixel_inventory import inventory_paths
from src.trackocd_cvpr.train_protocol_inventory import read_registered_known_metadata, support_summary

SOURCES = ('configs/trackocd_cvpr/cvpr2_train_pixel_inventory.json',
           'scripts/trackocd_cvpr/inventory_train_pixel_support.py',
           'src/trackocd_cvpr/train_pixel_inventory.py', 'src/trackocd_cvpr/train_protocol_inventory.py',
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
        raise ValueError('Exact current remote preregistration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{args.preregistration_commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered source changed')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve all attempts')
    started = time.monotonic()
    project_bytes = sum(path.stat().st_size for path in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if path.is_file())

    def guard():
        mem = {key: int(value.split()[0]) * 1024 for key, value in
               (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']:
            raise RuntimeError('RAM reserve')
        if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg['rss_bytes_ceiling']
                or time.monotonic() - started > cfg['wall_seconds_ceiling']
                or shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes']):
            raise RuntimeError('RSS/wall/free disk bound')
        output_bytes = sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) if out.exists() else 0
        if output_bytes > cfg['output_bytes_ceiling'] or project_bytes + output_bytes > safety['incremental_storage_hard_bytes']:
            raise RuntimeError('Small output/incremental bound')

    guard()
    annotation = Path(cfg['train_annotation'])
    if annotation.name != 'train.json' or sha256_file(annotation) != cfg['train_annotation_sha256']:
        raise ValueError('Only registered Train image/video metadata')
    out.mkdir(parents=True)
    launch = {'preregistration_commit': args.preregistration_commit,
              'source_sha256': {name: sha256_file(ROOT / name) for name in SOURCES},
              'old_reference_manifest_sha256': cfg['cache_manifest_sha256'],
              'Train_metadata_source_sha256': cfg['train_annotation_sha256']}
    atomic_json(out / 'launch.json', launch)
    try:
        rows, scope = read_registered_known_metadata(ROOT / cfg['cache_directory'], cfg['cache_manifest_sha256'])
        videos = {row['video_id'] for row in rows}
        if (len(rows), len(videos), scope['eligible_old_representation_fit_categories']) != (
                cfg['expected_safe_reference_tracks'], cfg['expected_safe_reference_videos'], cfg['expected_eligible_categories']):
            raise ValueError('Safe inventory universe changed')
        guard()
        metadata = image_video_metadata(annotation)
        guard()
        totals, complete = inventory_paths(metadata, videos, cfg['frame_root'], guard)
        support = support_summary(rows, complete)
        # Rename the generic intersection field to avoid implying new predictions.
        for row in support['per_category']:
            row['reference_video_intersection_with_complete_annotated_pixels'] = row.pop('reference_video_intersection_with_existing_predicted_cache')
        thresholds = support['category_video_support_threshold_counts']
        thresholds['reference_video_intersection_with_complete_annotated_pixels'] = thresholds.pop('reference_video_intersection_with_existing_predicted_cache')
        support.pop('predicted_cache_intersection_is_not_matching_or_predicted_track_denominator')
        result = {'status': 'COMPLETE_PATH_STAT_TRAIN_PIXEL_INVENTORY_NOT_FEATURE_OR_FOLD_ACCEPTANCE',
                  **launch, 'pixel_inventory': totals, 'reference_support': support, 'metadata_scope': scope,
                  'annotation_or_category_arrays_JSON_decoded': False, 'model_executed': False,
                  'Val_or_Test_access': False, 'optimization_steps': 0, 'interpretation': cfg['interpretation'],
                  'resources': {'wall_seconds': time.monotonic() - started,
                                'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                'training_GPU_hours': 0, 'disk_free_bytes': shutil.disk_usage(ROOT).free}}
        guard()
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'pixel_inventory': totals,
                          'resources': result['resources']}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error),
                                         'complete': False, 'wall_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
