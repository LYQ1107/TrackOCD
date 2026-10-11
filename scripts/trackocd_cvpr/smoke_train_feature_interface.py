#!/usr/bin/env python3
"""All-ID one-video frozen DINO engineering, no GT/semantic scores/training."""
from collections import defaultdict
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from scripts.trackocd_cvpr.run_train_physical_diagnostic import load as physical_plan, usage_guard
from src.trackocd_cvpr.train_physical_resume import explicit_reuse
from scripts.trackocd_cvpr.prepare_train_input_counterfactual import train_image
SOURCES = ('configs/trackocd_cvpr/cvpr0_train_feature_interface.json', 'configs/trackocd_core/common_features.json',
    'scripts/trackocd_cvpr/smoke_train_feature_interface.py', 'scripts/trackocd_cvpr/run_train_physical_diagnostic.py',
    'src/trackocd_cvpr/train_physical_resume.py', 'src/trackocd_cvpr/prefix_interface_checks.py',
    'scripts/trackocd_v2/build_common_features.py', 'src/trackocd_core/features.py',
    'src/trackocd_core/limited_masa_features.py', 'src/trackocd_core/physical_qualification.py',
    'configs/trackocd_cvpr/autonomous_research.json',
    'scripts/trackocd_core/run_masa_physical_qualification.py',
    'scripts/trackocd_core/smoke_masa_native.py',
    'scripts/trackocd_cvpr/smoke_train_physical_inputs.py',
    'scripts/trackocd_cvpr/prepare_train_input_counterfactual.py')
CONFIG = ROOT / SOURCES[0]


def load():
    cfg = json.loads(CONFIG.read_text())
    if sha256_file(ROOT / cfg['common_config']) != cfg['common_config_sha256'] or sha256_file(ROOT / cfg['physical_config']) != cfg['physical_config_sha256']:
        raise ValueError('Registered encoder/input configuration differs')
    physical, videos = physical_plan()
    if cfg['private_plan_sha256'] != physical['private_plan_sha256'] or cfg['frame_root'] != physical['frame_root']:
        raise ValueError('Wrong Train input plan')
    video = next(v for v in videos if v['video_id'] == cfg['video_id'])
    source = explicit_reuse(ROOT, video, physical)
    if source is None:
        raise ValueError('Only explicitly reused completed smoke video')
    return cfg, json.loads((ROOT / cfg['common_config']).read_text()), video, source


def worker():
    cfg, common, video, source = load()
    out = ROOT / cfg['output_directory']
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    from scripts.trackocd_core.smoke_masa_native import deny_network_and_external_children
    deny_network_and_external_children()
    def barrier(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            if ('/TAO-Amodal/annotations/' in path or '/frames/test/' in path or '/frames/val/' in path or
                path.endswith(('/roles.json', '/train_labels.parquet', '/evaluator_only_geometry_join.json'))):
                raise PermissionError('Predicted Train encoder rejects GT/roles/Val/Test')
    sys.addaudithook(barrier)
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from scripts.trackocd_v2.build_common_features import crop_box
    from src.trackocd_core.limited_masa_features import select_all_prefixes, LimitedMasaVideoCache
    from src.trackocd_cvpr.prefix_interface_checks import check_prefix_tails
    from scripts.trackocd_core.run_masa_physical_qualification import frozen_state_digest
    sys.path.insert(0, str(ROOT / common['local_repository']))
    from dinov2.hub.backbones import dinov2_vitb14
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(1027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = dinov2_vitb14(pretrained=False)
    state = torch.load(ROOT / common['checkpoint'], map_location='cpu', mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    before = frozen_state_digest(model)
    torch.cuda.set_per_process_memory_fraction(min(1., cfg['GPU_peak_bytes'] / torch.cuda.get_device_properties(0).total_memory))
    model.to('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    transform = transforms.Compose([transforms.Resize(tuple(common['resize']), interpolation=Image.Resampling.BILINEAR),
        transforms.ToTensor(), transforms.Normalize(common['normalization_mean'], common['normalization_std'])])
    source_path = ROOT / source['directory'] / 'shards' / source['record']['npz_filename']
    forwards, deduplicated = 0, 0
    with np.load(source_path, allow_pickle=False) as a:
        rows, count = select_all_prefixes(video, a, cfg['maximum_observations'])
        if (len(rows), count) != (cfg['expected_tracks'], cfg['expected_observations']):
            raise ValueError('All-ID interface universe differs')
        visual = np.full((count, 768), np.nan, np.float16)
        geometry, quality = np.empty((count, 4), np.float32), np.empty(count, np.float32)
        frames, images = np.empty(count, np.int64), np.empty(count, np.int64)
        grouped = defaultdict(list)
        for row in rows:
            for j, observation in enumerate(row['observations']):
                grouped[observation['image_position']].append((row['observation_offset'] + j, observation['source_offset']))
        for position, locations in sorted(grouped.items()):
            usage_guard(cfg, safety, out, started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            image = video['images'][position]
            path = train_image(Path(cfg['frame_root']), image['image_path'])
            if path.stat().st_size != image['image_bytes'] or sha256_file(path) != image['image_sha256']:
                raise ValueError('Registered current pixels changed')
            with Image.open(path) as raw:
                rgb = raw.convert('RGB')
                w, h = rgb.size
                unique = defaultdict(list)
                for offset, source_offset in locations:
                    geometry[offset] = np.clip(a['boxes'][source_offset] / [w, h, w, h], 0, 1)
                    quality[offset], frames[offset], images[offset] = a['score'][source_offset], image['frame_index'], image['image_id']
                    unique[tuple(a['boxes'][source_offset])].append(offset)
                boxes = list(unique)
                deduplicated += sum(len(v) - 1 for v in unique.values())
                for first in range(0, len(boxes), cfg['batch_size']):
                    usage_guard(cfg, safety, out, started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
                    if torch.cuda.max_memory_reserved() > cfg['GPU_peak_bytes'] or torch.cuda.mem_get_info()[0] < cfg['GPU_free_reserve_bytes']:
                        raise RuntimeError('Encoder GPU bound')
                    chosen = boxes[first:first + cfg['batch_size']]
                    tensors = [transform(crop_box(rgb, box, common['crop_context_fraction_per_side'])) for box in chosen]
                    with torch.inference_mode():
                        z = model.forward_features(torch.stack(tensors).to('cuda:0'))['x_norm_clstoken']
                        z = torch.nn.functional.normalize(z, dim=-1).cpu().numpy().astype(np.float16)
                    for box, value in zip(chosen, z):
                        for offset in unique[box]:
                            visual[offset] = value
                    forwards += 1
            atomic_json(out / 'progress.json', {'status': 'RUNNING_ALL_ID_TRAIN_DINO_INTERFACE',
                'current_image_position': position, 'batch_forwards': forwards, 'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - started})
    if not np.isfinite(visual).all() or frozen_state_digest(model) != before or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Incomplete or changed frozen encoder')
    offsets = np.asarray([row['observation_offset'] for row in rows] + [count], np.int64)
    checks = check_prefix_tails(visual, geometry, quality, frames, offsets)
    path = out / 'video_0047.all_id_prefix_features.npz'
    with path.with_suffix('.npz.tmp').open('xb') as writer:
        np.savez_compressed(writer, visual=visual, geometry=geometry, quality=quality, frame_index=frames, image_id=images,
                            track_id=np.asarray([row['physical_track_id'] for row in rows], np.int64), offsets=offsets)
        writer.flush(); os.fsync(writer.fileno())
    path.with_suffix('.npz.tmp').rename(path)
    cache = LimitedMasaVideoCache(path, video['video_id'])
    if len(cache.rows) != len(rows):
        raise ValueError('Real compact-reader interface mismatch')
    cache.close()
    usage_guard(cfg, safety, out, started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    result = {'status': 'PASS_ALL_ID_TRAIN_DINO_INTERFACE_NOT_MATCHED_SEMANTIC_RESULT',
        'config_sha256': sha256_file(CONFIG), 'common_config_sha256': cfg['common_config_sha256'],
        'DINO_sha256': common['checkpoint_sha256'], 'physical_npz_sha256': source['record']['npz_sha256'],
        'tracks': len(rows), 'observations': count, 'singletons': sum(len(r['observations']) == 1 for r in rows),
        'prefix_tail_mutation': checks, 'batch_forwards': forwards, 'deduplicated_identical_current_image_boxes': deduplicated,
        'private_feature_payload': {'filename': path.name, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)},
        'frozen_parameters_unchanged': True, 'GT_or_roles_or_text_or_Val_or_Test_input': False,
        'optimization_steps': 0, 'no_track_dropped_by_score_label_or_short_length': True,
        'future_physical_ID_does_not_enter_encoder_but_controller_protocol_still_unrepaired': True,
        'qualified_primary': False, 'matched_semantic_score_reported': False,
        'resources': {'wall_seconds': time.monotonic() - started, 'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'peak_GPU_reserved_bytes': torch.cuda.max_memory_reserved(), 'GPU_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'training_GPU_hours': 0}}
    atomic_json(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 'resources': result['resources']}), flush=True)


def parent(commit, proxy):
    cfg, common, video, source = load()
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    git = ['git'] + (['-c', f'http.proxy={proxy}'] if proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote = subprocess.check_output(git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"], text=True,
        timeout=35, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()[0]
    if remote != commit:
        raise ValueError('Exact remote interface registration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered interface source differs')
    repo = ROOT / common['local_repository']
    if (subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != common['upstream_commit']
            or subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
            or sha256_file(ROOT / common['checkpoint']) != common['checkpoint_sha256']):
        raise ValueError('Pinned DINO source/checkpoint differs')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve interface attempts')
    mem = {k: int(v.split()[0]) * 1024 for k, v in
           (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
    if mem['MemAvailable'] - cfg['RSS_bytes_ceiling'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']:
        raise RuntimeError('Projected RAM reserve')
    import shutil
    if shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes'] + cfg['output_bytes_ceiling']:
        raise RuntimeError('Projected disk reserve')
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'], text=True, timeout=10)
    from scripts.trackocd_cvpr.smoke_train_physical_inputs import eligible_uuid
    gpu = eligible_uuid(query, apps, cfg['GPU_peak_bytes'] + cfg['GPU_free_reserve_bytes'])
    out.mkdir(parents=True)
    atomic_json(out / 'launch.json', {'preregistration_commit': commit, 'gpu_uuid': gpu, 'fresh_query': query,
        'fresh_applications': apps, 'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES}})
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
                usage_guard(cfg, safety, out, started, owned_process_tree_rss(process.pid))
                time.sleep(2)
            if process.returncode != 0:
                raise RuntimeError('Owned encoder failed; preserve attempted payload/log')
        except BaseException as exc:
            error = str(exc)
            if process.poll() is None:
                process.terminate()
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
            p.error('remote-verified registration required')
        parent(args.preregistration_commit, args.git_proxy)
