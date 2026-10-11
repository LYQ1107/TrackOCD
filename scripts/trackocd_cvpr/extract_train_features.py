#!/usr/bin/env python3
"""Two bounded frozen Train workers; valid whole-video features are reused."""
import argparse
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
from src.trackocd_cvpr.train_feature_resume import assignments, recovered_feature
from scripts.trackocd_cvpr.run_train_physical_diagnostic import load as physical_plan, usage_guard
SOURCES = ('configs/trackocd_cvpr/cvpr0_train_features.json', 'scripts/trackocd_cvpr/extract_train_features.py',
    'src/trackocd_cvpr/train_feature_resume.py', 'src/trackocd_cvpr/train_frozen_encoder.py',
    'src/trackocd_cvpr/prefix_interface_checks.py', 'src/trackocd_core/limited_masa_features.py',
    'scripts/trackocd_cvpr/run_train_physical_diagnostic.py', 'scripts/trackocd_cvpr/prepare_train_input_counterfactual.py',
    'src/trackocd_cvpr/train_physical_resume.py', 'src/trackocd_core/physical_qualification.py',
    'configs/trackocd_cvpr/cvpr0_train_physical_diagnostic.json', 'configs/trackocd_cvpr/autonomous_research.json',
    'configs/trackocd_core/common_features.json', 'scripts/trackocd_v2/build_common_features.py',
    'scripts/trackocd_core/run_masa_physical_qualification.py', 'scripts/trackocd_core/smoke_masa_native.py',
    'scripts/trackocd_cvpr/smoke_train_physical_inputs.py', 'configs/trackocd_cvpr/cvpr0_train_feature_interface.json')
CONFIG = ROOT / SOURCES[0]


def load():
    cfg = json.loads(CONFIG.read_text())
    if sha256_file(ROOT / cfg['common_config']) != cfg['common_config_sha256'] or sha256_file(ROOT / cfg['physical_config']) != cfg['physical_config_sha256']:
        raise ValueError('Frozen encoder/physical configuration differs')
    common = json.loads((ROOT / cfg['common_config']).read_text())
    physical, videos = physical_plan()
    run = ROOT / cfg['physical_directory']
    if sha256_file(run / 'result.json') != cfg['physical_result_sha256'] or sha256_file(run / 'private_physical_manifest.json') != cfg['physical_manifest_sha256']:
        raise ValueError('Completed physical run/manifest binding differs')
    manifest = json.loads((run / 'private_physical_manifest.json').read_text())
    if not manifest['complete'] or len(videos) != cfg['expected_videos']:
        raise ValueError('Only complete full registered Train diagnostic universe')
    physical_records = {r['record']['video_id']: r for r in manifest['records']}
    if set(physical_records) != {v['video_id'] for v in videos}:
        raise ValueError('Physical video universe differs')
    reused = recovered_feature(ROOT, cfg, physical_records[cfg['reuse']['video_id']]['record'])
    return cfg, common, videos, physical_records, reused


def completed(cfg, video, physical):
    from src.trackocd_core.limited_masa_features import completed_feature, LimitedMasaVideoCache
    directory = ROOT / cfg['output_directory'] / 'shards'
    record = completed_feature(directory, video['video_id'], sha256_file(CONFIG), physical['record']['npz_sha256'])
    if record:
        reader = LimitedMasaVideoCache(directory / record['npz_filename'], video['video_id'])
        if len(reader.rows) != record['tracks'] or len(reader.data['visual']) != record['observations']:
            raise ValueError('Changed completed compact feature universe')
        reader.close()
    return record


def worker(index, launch_path):
    cfg, common, videos, physical, reused = load()
    launch = json.loads(Path(launch_path).read_text())
    if launch['source_sha256'] != {n: sha256_file(ROOT / n) for n in SOURCES}:
        raise ValueError('Owned feature source/config changed after launch')
    owned = set(launch['worker_videos'][str(index)])
    out = ROOT / cfg['output_directory']
    from scripts.trackocd_core.smoke_masa_native import deny_network_and_external_children
    deny_network_and_external_children()
    def barrier(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            if ('/TAO-Amodal/annotations/' in path or '/frames/test/' in path or '/frames/val/' in path or
                path.endswith(('/roles.json', '/train_labels.parquet', '/evaluator_only_geometry_join.json',
                               '/tiny_semantic_integration_v1/result.json'))):
                raise PermissionError('Train encoder rejects GT/roles/evaluator/Val/Test')
    sys.addaudithook(barrier)
    import numpy as np
    import torch
    from src.trackocd_core.limited_masa_features import LimitedMasaVideoCache
    from src.trackocd_cvpr.train_frozen_encoder import FrozenTrainEncoder
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    started = time.monotonic()
    def guard():
        usage_guard(cfg, safety, out, started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    guard()
    encoder = FrozenTrainEncoder(ROOT, cfg, common)
    records, new_observations = [], 0
    for video in videos:
        if video['video_id'] not in owned:
            continue
        guard()
        existing = completed(cfg, video, physical[video['video_id']])
        if existing:
            records.append(existing)
            continue
        source = physical[video['video_id']]
        source_path = ROOT / source['directory'] / 'shards' / source['record']['npz_filename']
        if sha256_file(source_path) != source['record']['npz_sha256']:
            raise ValueError('Completed physical input changed')
        arrays, statistics = encoder.video(video, source_path, guard, out / f'worker{index}_progress.json')
        filename = f"video_{video['video_id']:04d}.{launch['attempt']}.npz"
        path = out / 'shards' / filename
        temporary = path.with_suffix('.npz.tmp')
        if path.exists() or temporary.exists():
            raise ValueError('Never overwrite a retained feature attempt')
        with temporary.open('xb') as writer:
            np.savez_compressed(writer, **arrays)
            writer.flush(); os.fsync(writer.fileno())
        temporary.rename(path)
        reader = LimitedMasaVideoCache(path, video['video_id'])
        if len(reader.rows) != statistics['tracks']:
            raise ValueError('All-ID compact reader integration')
        reader.close()
        record = {'status': 'SEALED_COMPLETE_ALL_ID_PREFIX_FEATURES', 'video_id': video['video_id'], **statistics,
            'config_sha256': sha256_file(CONFIG), 'input_npz_sha256': source['record']['npz_sha256'],
            'npz_filename': filename, 'npz_bytes': path.stat().st_size, 'npz_sha256': sha256_file(path),
            'DINO_sha256': common['checkpoint_sha256'], 'source_sha256': launch['source_sha256'],
            'GT_or_roles_input': False, 'unmatched_tracks_removed': False}
        # Force native strict JSON before declaring an atomic completion marker.
        json.dumps(record, allow_nan=False)
        atomic_json(out / 'shards' / f"video_{video['video_id']:04d}.complete.json", record)
        records.append(record)
        new_observations += statistics['observations']
        del arrays
        atomic_json(out / f'worker{index}_summary.json', {'status': 'RUNNING_FROZEN_TRAIN_FEATURES',
            'videos_complete': len(records), 'new_observations': new_observations, 'last_video_id': video['video_id']})
    guard()
    atomic_json(out / f"worker{index}_{launch['attempt']}_done.json", {'status': 'COMPLETE_OWNED_FEATURE_PARTITION',
        'assigned_videos': len(owned), 'complete_videos': len(records), 'new_observations': new_observations,
        'resources': {'wall_seconds': time.monotonic() - started,
            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'peak_GPU_reserved_bytes': torch.cuda.max_memory_reserved(), 'GPU_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'training_GPU_hours': 0}})


def parent(commit, proxy, resume):
    cfg, common, videos, physical, reused = load()
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    git = ['git'] + (['-c', f'http.proxy={proxy}'] if proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote = subprocess.check_output(git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"], text=True,
        timeout=35, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()[0]
    if remote != commit:
        raise ValueError('Exact remote Train feature registration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered feature source differs')
    integration = ROOT / cfg['integration_result']
    if (sha256_file(integration) != cfg['integration_sha256'] or
            json.loads(integration.read_text())['status'] != 'PASS_TINY_ALL_ID_SEMANTIC_TRUNCATED_REFERENCE_INTEGRATION_NOT_MAIN_METRICS'):
        raise ValueError('Actual engineering integration required, never select by metrics')
    repo = ROOT / common['local_repository']
    if (subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != common['upstream_commit']
            or subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
            or sha256_file(ROOT / common['checkpoint']) != common['checkpoint_sha256']):
        raise ValueError('Existing pinned DINO repository/checkpoint differs')
    out = ROOT / cfg['output_directory']
    if (out.exists() and not resume) or (out / 'result.json').exists():
        raise ValueError('Preserve existing runs; resume only incomplete same registration')
    if out.exists() and json.loads((out / 'registration.json').read_text())['config_sha256'] != sha256_file(CONFIG):
        raise ValueError('Resume config differs')
    if shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes'] + cfg['output_bytes_ceiling']:
        raise RuntimeError('Projected free disk reserve')
    mem = {k: int(v.split()[0]) * 1024 for k, v in
           (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
    if mem['MemAvailable'] - cfg['parallel_workers'] * cfg['RSS_bytes_ceiling'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']:
        raise RuntimeError('RESOURCE_WAIT_PROJECTED_RAM_RESERVE')
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'], text=True, timeout=10)
    from scripts.trackocd_cvpr.smoke_train_physical_inputs import eligible_uuid
    remaining, gpus = query.splitlines(), []
    for _ in range(cfg['parallel_workers']):
        gpu = eligible_uuid('\n'.join(remaining), apps, cfg['GPU_peak_bytes'] + cfg['GPU_free_reserve_bytes'])
        gpus.append(gpu)
        remaining = [row for row in remaining if row.split(',')[0].strip() != gpu]
    if cfg['parallel_workers'] > safety['parallel_gpu_ceiling']:
        raise ValueError('Parallel GPU ceiling')
    if not out.exists():
        out.mkdir(parents=True)
        (out / 'shards').mkdir()
        atomic_json(out / 'registration.json', {'preregistration_commit': commit, 'config_sha256': sha256_file(CONFIG),
            'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES}})
    attempt = 'registered_' + format(time.time_ns(), 'x')
    launch_path = out / f'launch_{attempt}.json'
    launch = {'attempt': attempt, 'preregistration_commit': commit,
        'worker_videos': assignments(videos, cfg['reuse']['video_id'], cfg['parallel_workers']),
        'gpus': gpus, 'fresh_query': query, 'fresh_applications': apps,
        'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES}}
    atomic_json(launch_path, launch)
    processes, logs, error = [], [], None
    started = time.monotonic()
    from scripts.trackocd_core.smoke_masa_native import owned_process_tree_rss
    try:
        for index, gpu in enumerate(gpus):
            log = (out / f'private_worker{index}_{attempt}.log').open('x')
            logs.append(log)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'MPLCONFIGDIR': str(out / f'mpl{index}')}
            processes.append(subprocess.Popen([cfg['environment_python'], str(Path(__file__).resolve()), '--worker', str(index),
                '--launch', str(launch_path)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        parent_cfg = {**cfg, 'RSS_bytes_ceiling': cfg['RSS_bytes_ceiling'] * cfg['parallel_workers']}
        while any(p.poll() is None for p in processes):
            if any(p.returncode not in (None, 0) for p in processes):
                raise RuntimeError('Owned feature worker failed; retain all valid shards and logs')
            usage_guard(parent_cfg, safety, out, started, sum(owned_process_tree_rss(p.pid) for p in processes if p.poll() is None))
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('Owned feature partition failed')
    except BaseException as exc:
        error = str(exc)
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        raise
    finally:
        for log in logs:
            log.close()
        atomic_json(out / f'supervisor_{attempt}.json', {'owned_pids': [p.pid for p in processes],
            'returncodes': [p.returncode for p in processes], 'error': error,
            'wall_seconds': time.monotonic() - started, 'foreign_interference': False})
    records = [reused]
    for video in videos:
        if video['video_id'] == cfg['reuse']['video_id']:
            continue
        record = completed(cfg, video, physical[video['video_id']])
        if record is None:
            raise ValueError('Every registered video complete, no favorable omission')
        records.append({'directory': cfg['output_directory'] + '/shards', **record})
    if (len(records), sum(r['tracks'] for r in records), sum(r['observations'] for r in records)) != (
            cfg['expected_videos'], cfg['expected_tracks'], cfg['expected_observations']):
        raise ValueError('Full registered all-ID feature universe differs')
    manifest_path = out / 'private_feature_manifest.json'
    atomic_json(manifest_path, {'records': records, 'complete': True, 'config_sha256': sha256_file(CONFIG),
                              'physical_manifest_sha256': cfg['physical_manifest_sha256']})
    done = [json.loads((out / f'worker{i}_{attempt}_done.json').read_text()) for i in range(len(gpus))]
    result = {'status': 'COMPLETE_BOUNDED_TRAIN_ALL_ID_FEATURES_NOT_MAIN_QUALIFICATION',
        'preregistration_commit': commit, 'source_sha256': launch['source_sha256'], 'config_sha256': sha256_file(CONFIG),
        'private_manifest': {'filename': manifest_path.name, 'bytes': manifest_path.stat().st_size, 'sha256': sha256_file(manifest_path)},
        'videos': len(records), 'tracks': cfg['expected_tracks'], 'observations': cfg['expected_observations'],
        'reused_observations': reused['observations'], 'new_observations_current_attempt': sum(d['new_observations'] for d in done),
        'recovered_source_original_failure_preserved': True, 'valid_source_features_copied_or_reencoded': False,
        'no_short_unmatched_or_low_score_track_removed': True, 'GT_or_roles_or_Val_or_Test_encoder_input': False,
        'optimization_steps': 0, 'workers': done, 'qualified_primary': False, 'semantic_score_reported': False,
        'resources': {'supervisor_wall_seconds': time.monotonic() - started, 'training_GPU_hours': 0},
        'interpretation': cfg['interpretation']}
    atomic_json(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 'videos': len(records), 'observations': cfg['expected_observations'], 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker', type=int)
    p.add_argument('--launch')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preregistration-commit')
    p.add_argument('--git-proxy', default='')
    args = p.parse_args()
    if args.worker is not None:
        if not args.launch:
            p.error('owned launch assignment required')
        worker(args.worker, args.launch)
    else:
        if not args.preregistration_commit:
            p.error('remote-verified feature registration required')
        parent(args.preregistration_commit, args.git_proxy, args.resume)
