#!/usr/bin/env python3
"""One frozen worker, bounded Train inference, atomic shards and explicit reuse."""
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
from src.trackocd_cvpr.train_physical_resume import validate_plan, explicit_reuse
from scripts.trackocd_cvpr.prepare_train_input_counterfactual import train_image
SOURCES = ('configs/trackocd_cvpr/autonomous_research.json',
    'configs/trackocd_cvpr/cvpr0_train_physical_diagnostic.json',
    'src/trackocd_cvpr/train_physical_resume.py', 'scripts/trackocd_cvpr/run_train_physical_diagnostic.py',
    'scripts/trackocd_cvpr/smoke_train_physical_inputs.py',
    'scripts/trackocd_cvpr/prepare_train_input_counterfactual.py',
    'src/trackocd_core/masa_native.py', 'src/trackocd_core/physical_qualification.py')
CONFIG = ROOT / SOURCES[1]


def load():
    cfg = json.loads(CONFIG.read_text())
    path = ROOT / cfg['preflight_directory'] / 'private_image_plan.json'
    if sha256_file(path) != cfg['private_plan_sha256']:
        raise ValueError('Private Train image plan differs')
    return cfg, validate_plan(json.loads(path.read_text()), cfg)


def usage_guard(cfg, safety, out, started, rss_bytes):
    mem = {k: int(v.split()[0]) * 1024 for k, v in
           (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
    if mem['MemAvailable'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']:
        raise RuntimeError('RAM reserve')
    if rss_bytes > cfg['RSS_bytes_ceiling'] or time.monotonic() - started > cfg['wall_seconds_ceiling']:
        raise RuntimeError('Owned worker RSS/wall ceiling')
    if shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes']:
        raise RuntimeError('Disk reserve')
    if out.exists() and sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) > cfg['output_bytes_ceiling']:
        raise RuntimeError('Physical output bound')
    if sum(f.stat().st_size for f in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if f.is_file()) > safety['incremental_storage_hard_bytes']:
        raise RuntimeError('CVPR incremental storage hard ceiling')


def registered_records(cfg, videos, out):
    from src.trackocd_core.physical_qualification import completed_video
    records = []
    for video in videos:
        reused = explicit_reuse(ROOT, video, cfg)
        if reused:
            records.append(reused)
        else:
            record = completed_video(out, video, sha256_file(CONFIG))
            if record:
                records.append({'directory': cfg['output_directory'], 'record': record,
                                'reused_without_copy_or_forward': False})
    return records


def worker(attempt):
    started = time.monotonic()
    cfg, videos = load()
    safety = json.loads((ROOT / SOURCES[0]).read_text())
    out = ROOT / cfg['output_directory']
    from scripts.trackocd_core.smoke_masa_native import deny_network_and_external_children
    deny_network_and_external_children()
    def barrier(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            if ('/TAO-Amodal/annotations/' in path or '/frames/test/' in path or '/frames/val/' in path or
                path.endswith(('/roles.json', '/train_labels.parquet', '/evaluator_only_geometry_join.json'))):
                raise PermissionError('Frozen physical worker rejects labels/Val/Test')
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
    records = registered_records(cfg, videos, out)
    finished = {r['record']['video_id'] for r in records}
    model, proof = build_frozen_model()
    if proof['checkpoint_sha256'] != cfg['checkpoint_sha256'] or json_digest(proof['native_model_config']) != cfg['native_model_config_sha256']:
        raise ValueError('Native model identity differs')
    state_before = frozen_state_digest(model)
    torch.cuda.set_per_process_memory_fraction(min(1., cfg['GPU_peak_bytes'] / torch.cuda.get_device_properties(0).total_memory))
    model.to('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    forwards = 0
    for video in videos:
        if video['video_id'] in finished:
            continue
        model.tracker.reset()
        arrays = {'image_id': [], 'frame_index': [], 'frame_offsets': [0], 'track_id': [], 'boxes': [], 'score': [],
                  'det_offsets': [0], 'det_boxes': [], 'det_score': []}
        for ordinal, image in enumerate(video['images']):
            usage_guard(cfg, safety, out, started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            if (forwards >= cfg['new_forward_ceiling'] or torch.cuda.max_memory_reserved() > cfg['GPU_peak_bytes']
                    or torch.cuda.mem_get_info()[0] < cfg['GPU_free_reserve_bytes']):
                raise RuntimeError('Forward/GPU bound')
            path = train_image(Path(cfg['frame_root']), image['image_path'])
            if path.stat().st_size != image['image_bytes'] or sha256_file(path) != image['image_sha256']:
                raise ValueError('Registered current Train pixels changed')
            with Image.open(path) as raw:
                pixels = np.asarray(raw.convert('RGB'), np.uint8).copy()
            detections, tracks = infer_current_frame(model, pixels, ordinal, 'cuda:0')
            torch.cuda.synchronize()
            forwards += 1
            arrays['image_id'].append(image['image_id'])
            arrays['frame_index'].append(image['frame_index'])
            arrays['track_id'].extend(tracks.instances_id.cpu().numpy().tolist())
            arrays['boxes'].extend(tracks.bboxes.cpu().numpy().tolist())
            arrays['score'].extend(tracks.scores.cpu().numpy().tolist())
            arrays['frame_offsets'].append(len(arrays['track_id']))
            arrays['det_boxes'].extend(detections.bboxes.cpu().numpy().tolist())
            arrays['det_score'].extend(detections.scores.cpu().numpy().tolist())
            arrays['det_offsets'].append(len(arrays['det_score']))
            atomic_json(out / 'progress.json', {'status': 'RUNNING_TRAIN_PHYSICAL_DIAGNOSTIC',
                'completed_videos': len(records), 'registered_videos': len(videos), 'current_video': video['video_id'],
                'current_video_frames': ordinal + 1, 'current_invocation_forwards': forwards,
                'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - started})
        if frozen_state_digest(model) != state_before or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Model parameters changed')
        arrays = {k: np.asarray(v, np.float32 if k in ('boxes', 'score', 'det_boxes', 'det_score') else np.int64)
                  for k, v in arrays.items()}
        arrays['boxes'] = arrays['boxes'].reshape(-1, 4)
        arrays['det_boxes'] = arrays['det_boxes'].reshape(-1, 4)
        record = seal_video(out, video, arrays, sha256_file(CONFIG), attempt, {'frozen_state_unchanged': True})
        records.append({'directory': cfg['output_directory'], 'record': record, 'reused_without_copy_or_forward': False})
        atomic_json(out / 'private_physical_manifest.json', {'records': records, 'complete': False})
    usage_guard(cfg, safety, out, started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    checked = registered_records(cfg, videos, out)
    if len(checked) != cfg['videos'] or sum(r['record']['images'] for r in checked) != cfg['canonical_images']:
        raise ValueError('Incomplete full registered diagnostic universe')
    manifest_path = out / 'private_physical_manifest.json'
    atomic_json(manifest_path, {'records': checked, 'complete': True, 'private_plan_sha256': cfg['private_plan_sha256'],
                              'config_sha256': sha256_file(CONFIG)})
    physical_ids, prediction_rows, detection_rows, singletons = 0, 0, 0, 0
    for row in checked:
        with np.load(ROOT / row['directory'] / 'shards' / row['record']['npz_filename'], allow_pickle=False) as a:
            _, lengths = np.unique(a['track_id'], return_counts=True)
            physical_ids += len(lengths)
            singletons += int((lengths == 1).sum())
        prediction_rows += row['record']['prediction_rows']
        detection_rows += row['record']['detection_rows']
    result = {'status': 'COMPLETE_BOUNDED_TRAIN_PHYSICAL_INPUT_NOT_MAIN_QUALIFICATION',
        'config_sha256': sha256_file(CONFIG), 'private_plan_sha256': cfg['private_plan_sha256'],
        'native_checkpoint_sha256': cfg['checkpoint_sha256'], 'native_model_config_sha256': cfg['native_model_config_sha256'],
        'private_manifest': {'filename': manifest_path.name, 'sha256': sha256_file(manifest_path), 'bytes': manifest_path.stat().st_size},
        'videos': len(checked), 'canonical_images': cfg['canonical_images'],
        'current_invocation_new_forwards': forwards, 'explicit_reused_source_videos': 1,
        'source_shard_copied_or_reextracted': False, 'physical_ids': physical_ids, 'prediction_rows': prediction_rows,
        'detection_rows': detection_rows, 'singleton_physical_ids': singletons,
        'cadence': 'Train annotation metadata images, not dense frame-online',
        'frozen_state_unchanged': True, 'GT_or_text_or_Val_or_Test_input': False, 'optimizer_steps': 0,
        'frontend_provenance': cfg['exact_release_stage_data_binding'], 'primary_frontend_qualified': False,
        'GT_reference': cfg['reference'], 'semantic_score_reported': False,
        'resources': {'wall_seconds': time.monotonic() - started, 'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'peak_GPU_reserved_bytes': torch.cuda.max_memory_reserved(), 'GPU_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'training_GPU_hours': 0}}
    atomic_json(out / 'result.json', result)
    atomic_json(out / 'progress.json', {'status': result['status'], 'completed_videos': len(checked), 'current_invocation_new_forwards': forwards})
    print(json.dumps({'status': result['status'], 'resources': result['resources']}), flush=True)


def parent(commit, proxy, resume):
    cfg, videos = load()
    safety = json.loads((ROOT / SOURCES[0]).read_text())
    git = ['git'] + (['-c', f'http.proxy={proxy}'] if proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote = subprocess.check_output(git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"], text=True,
        timeout=35, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()[0]
    if remote != commit:
        raise ValueError('Exact remote physical diagnostic preregistration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered source differs')
    out = ROOT / cfg['output_directory']
    if (out.exists() and not resume) or (out / 'result.json').exists():
        raise ValueError('Preserve existing attempt; resume only an incomplete registered run')
    records = registered_records(cfg, videos, out)
    if len(records) == cfg['videos']:
        raise ValueError('All valid physical shards already exist; do not request new inference')
    if out.exists() and json.loads((out / 'registration.json').read_text())['config_sha256'] != sha256_file(CONFIG):
        raise ValueError('Resume configuration changed')
    if shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes'] + cfg['subsequent_physical_plus_features_budget_bytes']:
        raise RuntimeError('Disk projected reserve')
    mem = {k: int(v.split()[0]) * 1024 for k, v in
           (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
    if mem['MemAvailable'] - cfg['RSS_bytes_ceiling'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']:
        raise RuntimeError('RESOURCE_WAIT_PROJECTED_RAM_RESERVE')
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'], text=True, timeout=10)
    from scripts.trackocd_cvpr.smoke_train_physical_inputs import eligible_uuid
    gpu = eligible_uuid(query, apps, cfg['GPU_peak_bytes'] + cfg['GPU_free_reserve_bytes'])
    if not out.exists():
        out.mkdir(parents=True)
        (out / 'shards').mkdir()
        atomic_json(out / 'registration.json', {'preregistration_commit': commit, 'config_sha256': sha256_file(CONFIG),
            'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES}})
    attempt = 'registered_' + format(time.time_ns(), 'x')
    atomic_json(out / f'launch_{attempt}.json', {'preregistration_commit': commit, 'gpu_uuid': gpu,
        'fresh_query': query, 'fresh_applications': apps, 'already_valid_videos': len(records), 'resume': resume})
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'MPLCONFIGDIR': str(out / 'mpl')}
    started = time.monotonic()
    with (out / f'private_worker_{attempt}.log').open('x') as log:
        process = subprocess.Popen([cfg['environment_python'], str(Path(__file__).resolve()), '--worker', '--attempt', attempt],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        from scripts.trackocd_core.smoke_masa_native import owned_process_tree_rss
        error = None
        try:
            while process.poll() is None:
                usage_guard(cfg, safety, out, started, owned_process_tree_rss(process.pid))
                time.sleep(2)
            if process.returncode != 0:
                raise RuntimeError('Owned physical diagnostic failed; preserve logs and completed shards')
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
            atomic_json(out / f'supervisor_{attempt}.json', {'owned_pid': process.pid, 'returncode': process.returncode,
                'error': error, 'wall_seconds': time.monotonic() - started, 'foreign_interference': False})
    result = json.loads((out / 'result.json').read_text())
    print(json.dumps({'status': result['status'], 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--attempt')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preregistration-commit')
    p.add_argument('--git-proxy', default='')
    args = p.parse_args()
    if args.worker:
        if not args.attempt:
            p.error('owned attempt identity required')
        worker(args.attempt)
    else:
        if not args.preregistration_commit:
            p.error('remote-verified registration required')
        parent(args.preregistration_commit, args.git_proxy, args.resume)
