#!/usr/bin/env python3
"""Fresh one-GPU frozen Train smoke, no GT/Val/Test or full-data launch."""
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
from scripts.trackocd_cvpr.prepare_train_input_counterfactual import train_image
SOURCES = ('configs/trackocd_cvpr/cvpr0_train_physical_smoke.json',
           'scripts/trackocd_cvpr/smoke_train_physical_inputs.py',
           'scripts/trackocd_cvpr/prepare_train_input_counterfactual.py',
           'src/trackocd_core/masa_native.py', 'src/trackocd_core/physical_qualification.py')
CONFIG = ROOT / SOURCES[0]


def eligible_uuid(query, applications, required_free_bytes):
    busy = set(line.split(',')[0].strip() for line in applications.splitlines())
    for row in query.splitlines():
        gpu, used, free, util = [s.strip() for s in row.split(',')]
        if gpu in busy or any('N/A' in x for x in (used, free, util)):
            continue
        if int(used) < 100 and int(util) == 0 and int(free) * 1024**2 >= required_free_bytes:
            return gpu
    raise RuntimeError('RESOURCE_WAIT_NO_FRESH_IDLE_GPU')


def load():
    cfg = json.loads(CONFIG.read_text())
    p = ROOT / cfg['preflight_directory'] / 'private_image_plan.json'
    if sha256_file(p) != cfg['private_plan_sha256']:
        raise ValueError('Registered metadata-only Train plan changed')
    plan = json.loads(p.read_text())
    if plan['GT_or_category_fields'] is not False or len(plan['videos']) != 57:
        raise ValueError('Wrong input plan')
    video = plan['videos'][0]
    if len(video['images']) != cfg['canonical_images']:
        raise ValueError('Smoke video/image universe differs')
    allowed = {'image_id', 'frame_index', 'image_path', 'image_bytes', 'image_sha256'}
    if any(set(i) != allowed for i in video['images']):
        raise ValueError('Physical input has unexpected metadata/label fields')
    return cfg, video


def worker():
    started = time.monotonic()
    cfg, video = load()
    out = ROOT / cfg['output_directory']
    from scripts.trackocd_core.smoke_masa_native import deny_network_and_external_children, snapshot_instances
    deny_network_and_external_children()
    def barrier(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            if ('/TAO-Amodal/annotations/' in path or '/frames/test/' in path or '/frames/val/' in path or
                path.endswith(('/roles.json', '/train_labels.parquet', '/evaluator_only_geometry_join.json'))):
                raise PermissionError('Frozen physical predictor rejects labels/Val/Test')
    sys.addaudithook(barrier)
    import numpy as np
    import torch
    from PIL import Image
    from src.trackocd_core.masa_native import build_frozen_model, infer_current_frame
    from src.trackocd_core.physical_qualification import json_digest, seal_video
    from scripts.trackocd_core.run_masa_physical_qualification import frozen_state_digest
    torch.set_num_threads(1)
    torch.manual_seed(1027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, proof = build_frozen_model()
    if proof['checkpoint_sha256'] != cfg['checkpoint_sha256'] or json_digest(proof['native_model_config']) != cfg['native_model_config_sha256']:
        raise ValueError('Frozen model identity differs')
    state_before = frozen_state_digest(model)
    torch.cuda.set_per_process_memory_fraction(min(1., cfg['GPU_peak_bytes'] / torch.cuda.get_device_properties(0).total_memory))
    model.to('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    actual_forwards = 0

    def guard():
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable'] < .25 * mem['MemTotal'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg['RSS_bytes_ceiling']:
            raise RuntimeError('RAM bound')
        if time.monotonic() - started > cfg['wall_seconds_ceiling'] or torch.cuda.max_memory_reserved() > cfg['GPU_peak_bytes']:
            raise RuntimeError('Wall/GPU peak bound')
        if torch.cuda.mem_get_info()[0] < cfg['GPU_free_reserve_bytes']:
            raise RuntimeError('GPU free reserve')
        if actual_forwards > cfg['forward_ceiling']:
            raise RuntimeError('Forward ceiling')

    def current(image):
        path = train_image(Path(cfg['frame_root']), image['image_path'])
        if path.stat().st_size != image['image_bytes'] or sha256_file(path) != image['image_sha256']:
            raise ValueError('Registered Train pixels changed')
        with Image.open(path) as raw:
            return np.asarray(raw.convert('RGB'), dtype=np.uint8).copy()

    arrays = {'image_id': [], 'frame_index': [], 'frame_offsets': [0], 'track_id': [], 'boxes': [], 'score': [],
              'det_offsets': [0], 'det_boxes': [], 'det_score': []}
    snapshots = []
    model.tracker.reset()
    for ordinal, image in enumerate(video['images']):
        guard()
        detections, tracks = infer_current_frame(model, current(image), ordinal, 'cuda:0')
        torch.cuda.synchronize()
        actual_forwards += 1
        snapshots.append({'detections': snapshot_instances(detections, False), 'tracks': snapshot_instances(tracks, True)})
        arrays['image_id'].append(image['image_id'])
        arrays['frame_index'].append(image['frame_index'])
        arrays['track_id'].extend(tracks.instances_id.cpu().numpy().tolist())
        arrays['boxes'].extend(tracks.bboxes.cpu().numpy().tolist())
        arrays['score'].extend(tracks.scores.cpu().numpy().tolist())
        arrays['frame_offsets'].append(len(arrays['track_id']))
        arrays['det_boxes'].extend(detections.bboxes.cpu().numpy().tolist())
        arrays['det_score'].extend(detections.scores.cpu().numpy().tolist())
        arrays['det_offsets'].append(len(arrays['det_score']))
        atomic_json(out / 'progress.json', {'status': 'RUNNING_CANONICAL_SMOKE', 'canonical_frames': ordinal + 1,
                    'actual_forwards': actual_forwards, 'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - started})
    model.tracker.reset()
    changed_future = []
    pixel_changes = []
    for ordinal, image in enumerate(video['images'][:4]):
        guard()
        pixels = current(image)
        before_pixels = hashlib.sha256(pixels.tobytes()).hexdigest()
        if ordinal >= 2:
            pixels = 255 - pixels
        pixel_changes.append(hashlib.sha256(pixels.tobytes()).hexdigest() != before_pixels)
        detections, tracks = infer_current_frame(model, pixels, ordinal, 'cuda:0')
        torch.cuda.synchronize()
        actual_forwards += 1
        changed_future.append({'detections': snapshot_instances(detections, False), 'tracks': snapshot_instances(tracks, True)})
    prefix_equal = all(snapshots[i][k]['array_sha256'] == changed_future[i][k]['array_sha256']
                       for i in (0, 1) for k in ('detections', 'tracks'))
    if not prefix_equal or pixel_changes != [False, False, True, True] or frozen_state_digest(model) != state_before or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Changed-future or frozen-parameter smoke failed')
    guard()
    arrays = {k: np.asarray(v, np.float32 if k in ('boxes', 'score', 'det_boxes', 'det_score') else np.int64)
              for k, v in arrays.items()}
    arrays['boxes'] = arrays['boxes'].reshape(-1, 4)
    arrays['det_boxes'] = arrays['det_boxes'].reshape(-1, 4)
    (out / 'shards').mkdir()
    record = seal_video(out, video, arrays, sha256_file(CONFIG), 'registered_smoke', {'frozen_state_unchanged': True})
    result = {'status': 'PASS_BOUNDED_TRAIN_PHYSICAL_SMOKE_NOT_MAIN_QUALIFICATION',
        'config_sha256': sha256_file(CONFIG), 'private_plan_sha256': cfg['private_plan_sha256'],
        'native_checkpoint_sha256': cfg['checkpoint_sha256'], 'native_model_config_sha256': cfg['native_model_config_sha256'],
        'actual_forwards': actual_forwards, 'canonical_images': len(video['images']), 'videos': 1,
        'shard': record, 'changed_future_prefix_equal': prefix_equal,
        'pixel_changes_actually_hashed': pixel_changes, 'frozen_state_unchanged': True,
        'scope': 'Train annotated-cadence physical engineering only; incomplete release binding; not clean primary or end-to-end score',
        'GT_or_text_or_Val_or_Test_input': False, 'optimizer_steps': 0,
        'resources': {'wall_seconds': time.monotonic() - started, 'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'peak_GPU_reserved_bytes': torch.cuda.max_memory_reserved(), 'GPU_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
            'training_GPU_hours': 0}}
    atomic_json(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 'resources': result['resources']}), flush=True)


def parent(commit, proxy):
    cfg, _ = load()
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    git = ['git'] + (['-c', f'http.proxy={proxy}'] if proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote = subprocess.check_output(git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"], text=True,
                                     timeout=35, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()[0]
    if remote != commit:
        raise ValueError('Exact remote smoke preregistration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered source differs')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve actual smoke attempts')
    if shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes'] + cfg['output_bytes_ceiling']:
        raise RuntimeError('Disk reserve')
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'], text=True, timeout=10)
    gpu = eligible_uuid(query, apps, cfg['GPU_peak_bytes'] + cfg['GPU_free_reserve_bytes'])
    mem = {k: int(v.split()[0]) * 1024 for k, v in
           (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
    if mem['MemAvailable'] - cfg['RSS_bytes_ceiling'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']:
        raise RuntimeError('RESOURCE_WAIT_PROJECTED_RAM_RESERVE')
    out.mkdir(parents=True)
    atomic_json(out / 'launch.json', {'preregistration_commit': commit, 'gpu_uuid': gpu, 'fresh_query': query,
                                    'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES}})
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'MPLCONFIGDIR': str(out / 'mpl')}
    started = time.monotonic()
    with (out / 'private_worker.log').open('x') as log:
        process = subprocess.Popen([cfg['environment_python'], str(Path(__file__).resolve()), '--worker'],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        from scripts.trackocd_core.smoke_masa_native import owned_process_tree_rss
        error = None
        try:
            while process.poll() is None:
                mem = {k: int(v.split()[0]) * 1024 for k, v in
                       (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
                if mem['MemAvailable'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']:
                    raise RuntimeError('RAM reserve')
                if owned_process_tree_rss(process.pid) > cfg['RSS_bytes_ceiling'] or time.monotonic() - started > cfg['wall_seconds_ceiling']:
                    raise RuntimeError('Owned worker resource ceiling')
                if shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes']:
                    raise RuntimeError('Free disk reserve')
                if sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) > cfg['output_bytes_ceiling']:
                    raise RuntimeError('Smoke output bound')
                time.sleep(2)
            if process.returncode != 0:
                raise RuntimeError('Owned smoke worker failed; preserve private log')
        except BaseException as exc:
            error = str(exc)
            if process.poll() is None:
                process.terminate()  # Only this directly spawned child, never historical/foreign PID.
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            raise
        finally:
            atomic_json(out / 'supervisor.json', {'owned_pid': process.pid, 'returncode': process.returncode,
                'error': error, 'wall_seconds': time.monotonic() - started, 'foreign_interference': False})
    result = json.loads((out / 'result.json').read_text())
    print(json.dumps({'status': result['status'], 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--preregistration-commit')
    p.add_argument('--git-proxy', default='')
    args = p.parse_args()
    if args.worker:
        worker()
    else:
        if not args.preregistration_commit:
            p.error('remote-verified preregistration commit required')
        parent(args.preregistration_commit, args.git_proxy)
