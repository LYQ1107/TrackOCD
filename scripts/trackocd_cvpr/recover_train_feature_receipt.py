#!/usr/bin/env python3
"""Validate retained all-ID pixels/metadata/features; no encoder or GT read."""
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
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
SOURCES = ('configs/trackocd_cvpr/cvpr0_train_feature_recovery.json',
           'scripts/trackocd_cvpr/recover_train_feature_receipt.py',
           'src/trackocd_cvpr/prefix_interface_checks.py')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preregistration-commit', required=True)
    p.add_argument('--git-proxy', default='')
    args = p.parse_args()
    cfg = json.loads((ROOT / SOURCES[0]).read_text())
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    git = ['git'] + (['-c', f'http.proxy={args.git_proxy}'] if args.git_proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote = subprocess.check_output(git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"], text=True,
        timeout=35, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()[0]
    if remote != args.preregistration_commit:
        raise ValueError('Exact remote recovery registration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{remote}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered recovery source changed')
    out, failed = ROOT / cfg['output_directory'], ROOT / cfg['failed_directory']
    if out.exists() or (failed / 'result.json').exists():
        raise ValueError('Preserve original attempt; only recover incomplete receipt into new namespace')
    started = time.monotonic()
    for name, digest in (('launch.json', cfg['launch_sha256']), ('private_worker.log', cfg['private_log_sha256']),
                         (cfg['feature_filename'], cfg['feature_sha256'])):
        if sha256_file(failed / name) != digest:
            raise ValueError('Retained failed attempt or completed payload changed')
    launch = json.loads((failed / 'launch.json').read_text())
    if launch['preregistration_commit'] != cfg['failed_preregistration_commit']:
        raise ValueError('Original registration differs')
    for name, digest in launch['source_sha256'].items():
        archived = subprocess.check_output(['git', 'show', f"{cfg['failed_preregistration_commit']}:{name}"], cwd=ROOT)
        if hashlib.sha256(archived).hexdigest() != digest:
            raise ValueError('Original launch not bound to actual registered source')
    supervisor = json.loads((failed / 'supervisor.json').read_text())
    log = (failed / 'private_worker.log').read_text()
    if supervisor['returncode'] != cfg['original_exit_code'] or 'TypeError: Object of type int64 is not JSON serializable' not in log or "atomic_json(out / 'result.json', result)" not in log:
        raise ValueError('Only the verified final JSON-serialization failure is recoverable here')
    import numpy as np
    from PIL import Image
    from scripts.trackocd_cvpr.smoke_train_feature_interface import load
    from scripts.trackocd_cvpr.prepare_train_input_counterfactual import train_image
    from src.trackocd_core.limited_masa_features import select_all_prefixes, LimitedMasaVideoCache
    from src.trackocd_cvpr.prefix_interface_checks import check_prefix_tails
    interface, common, video, source = load()
    if sha256_file(ROOT / common['checkpoint']) != common['checkpoint_sha256']:
        raise ValueError('Existing frozen DINO weights changed')
    feature_path = failed / cfg['feature_filename']
    if feature_path.stat().st_size != cfg['feature_bytes']:
        raise ValueError('Feature bytes differ')
    with np.load(ROOT / source['directory'] / 'shards' / source['record']['npz_filename'], allow_pickle=False) as a, np.load(feature_path, allow_pickle=False) as z:
        rows, count = select_all_prefixes(video, a, interface['maximum_observations'])
        if (len(rows), count) != (cfg['expected_tracks'], cfg['expected_observations']):
            raise ValueError('Original all-ID universe differs')
        if set(z.files) != {'visual', 'geometry', 'quality', 'frame_index', 'image_id', 'track_id', 'offsets'}:
            raise ValueError('Unexpected feature/GT fields')
        np.testing.assert_array_equal(z['track_id'], [r['physical_track_id'] for r in rows])
        np.testing.assert_array_equal(z['offsets'], [r['observation_offset'] for r in rows] + [count])
        sizes = {}
        for position, image in enumerate(video['images']):
            path = train_image(Path(interface['frame_root']), image['image_path'])
            if path.stat().st_size != image['image_bytes'] or sha256_file(path) != image['image_sha256']:
                raise ValueError('Original pixels changed')
            with Image.open(path) as raw:
                sizes[position] = raw.size
        for row in rows:
            for j, observation in enumerate(row['observations']):
                offset = row['observation_offset'] + j
                position, source_offset = observation['image_position'], observation['source_offset']
                image = video['images'][position]
                w, h = sizes[position]
                geometry = np.clip(a['boxes'][source_offset] / [w, h, w, h], 0, 1).astype(np.float32)
                np.testing.assert_array_equal(z['geometry'][offset], geometry)
                if (z['quality'][offset] != a['score'][source_offset] or z['frame_index'][offset] != image['frame_index']
                        or z['image_id'][offset] != image['image_id']):
                    raise ValueError('Feature metadata/quality differs from actual physical rows')
        norms = np.linalg.norm(z['visual'].astype(np.float32), axis=1)
        if not np.isfinite(norms).all() or np.max(np.abs(norms - 1)) > .002:
            raise ValueError('Invalid stored L2-normalized FP16 descriptors')
        checks = check_prefix_tails(z['visual'], z['geometry'], z['quality'], z['frame_index'], z['offsets'])
    cache = LimitedMasaVideoCache(feature_path, video['video_id'])
    if len(cache.rows) != cfg['expected_tracks']:
        raise ValueError('Compact interface invalid')
    cache.close()
    mem = {k: int(v.split()[0]) * 1024 for k, v in
           (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
    if (mem['MemAvailable'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg['rss_bytes_ceiling']
            or time.monotonic() - started > cfg['wall_seconds_ceiling']
            or shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes']):
        raise RuntimeError('CPU recovery resource bounds')
    out.mkdir(parents=True)
    result = {'status': 'RECOVERED_VALID_ALL_ID_FEATURE_INPUT_AFTER_FAILED_FINAL_JSON_NOT_NORMAL_WORKER_SUCCESS',
        'preregistration_commit': remote, 'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES},
        'original_preregistration_commit': cfg['failed_preregistration_commit'], 'original_worker_returncode': supervisor['returncode'],
        'original_supervisor_wall_seconds': supervisor['wall_seconds'], 'original_GPU_RSS_peaks': 'NOT_RETAINED_NOT_FABRICATED',
        'original_frozen_parameter_guard': 'Registered failing code path passes frozen guard before writing NPZ; hash value not retained',
        'original_final_progress': json.loads((failed / 'progress.json').read_text()),
        'private_feature_payload': {'directory': cfg['failed_directory'], 'filename': cfg['feature_filename'],
            'bytes': cfg['feature_bytes'], 'sha256': cfg['feature_sha256']},
        'private_failed_log_sha256': cfg['private_log_sha256'], 'original_launch_sha256': cfg['launch_sha256'],
        'physical_npz_sha256': source['record']['npz_sha256'], 'DINO_sha256': common['checkpoint_sha256'],
        'common_config_sha256': interface['common_config_sha256'],
        'tracks': cfg['expected_tracks'], 'observations': cfg['expected_observations'], 'singletons': 86,
        'all_original_source_and_current_pixel_metadata_geometry_quality_L2_reader_verified': True,
        'prefix_tail_mutation': checks, 'new_encoder_forward': False, 'payload_copied_or_rewritten': False,
        'optimization_steps': 0, 'new_GT_or_Val_or_Test_access': False, 'semantic_score_reported': False,
        'frontend_primary_qualified': False, 'interpretation': cfg['interpretation'],
        'resources': {'wall_seconds': time.monotonic() - started,
            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'training_GPU_hours': 0, 'GPU_used': False}}
    atomic_json(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    main()
