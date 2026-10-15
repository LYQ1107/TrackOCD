#!/usr/bin/env python3
"""Bounded independent-pseudo-Novel Train cache pilot with a fresh source namespace."""
import argparse
import json
import mmap
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration
from src.trackocd_cvpr.scoped_train_census import top_ranges, object_ranges, array_ranges, integer_value
from src.trackocd_cvpr.GT_video_cache import (selected_video_plans, write_atomic_shard,
                                              read_verified_shard, check_every_prefix)
from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory
from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest, verified_encoder_sources

CONFIG = 'configs/trackocd_cvpr/independent_final_GT_video_cache_pilot_v1.json'
SOURCES = (CONFIG, 'scripts/trackocd_cvpr/independent_final_GT_video_cache.py',
           'src/trackocd_cvpr/GT_video_cache.py', 'scripts/trackocd_cvpr/audit_joint_known_protocol.py',
           'scripts/trackocd_cvpr/smoke_selected_frozen_encoder.py',
           'scripts/trackocd_cvpr/audit_train_metadata_containment.py',
           'src/trackocd_cvpr/remote_registration.py', 'src/trackocd_cvpr/scoped_train_census.py',
           'src/trackocd_cvpr/metadata_only.py', 'src/trackocd_v2/io.py',
           'src/trackocd_core/features.py', 'scripts/trackocd_v2/build_common_features.py',
           'configs/trackocd_core/common_features.json',
           'configs/trackocd_cvpr/formal_research_scope_approval_v1.json',
           'configs/trackocd_cvpr/frozen_global_data_GT_control_v1.json',
           'configs/trackocd_cvpr/autonomous_research.json')


def assigned_video_ids(path, role, *, expected_count):
    """Project only opaque IDs for one frozen role; class fields are never decoded."""
    if type(role) is not str or not role:
        raise ValueError('Explicit non-empty assignment role')
    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        assignment = object_ranges(data, *top_ranges(data)['assignment'])
        roles = object_ranges(data, *assignment['video_ids'])
        if role not in roles:
            raise ValueError('Unknown assignment role')
        ids = [integer_value(data, span) for span in array_ranges(data, *roles[role])]
    if (type(expected_count) is not int or expected_count < 1
            or len(ids) != expected_count or len(set(ids)) != expected_count
            or any(type(value) is not int or value < 0 for value in ids)):
        raise ValueError('Exact fixed unique opaque video-id role')
    return tuple(ids)


def require_idle_gpu(cfg):
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        timeout=10).decode().splitlines()
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], timeout=10).decode().splitlines()
    occupied = {row.strip() for row in apps}
    wanted = cfg['GPU_uuid']
    if wanted != 'FIRST_FRESH_IDLE_UUID':
        raise ValueError('Independent pilot must choose a currently bound or first fresh idle UUID')
    eligible = []
    for row in rows:
        fields = [value.strip() for value in row.split(',')]
        if len(fields) != 3 or fields[0] in occupied:
            continue
        try:
            used, utilization = map(int, fields[1:])
        except ValueError:
            continue
        if 0 <= used <= cfg['GPU_idle_max_used_MiB'] and 0 <= utilization <= cfg['GPU_idle_max_utilization_percent']:
            eligible.append(fields[0])
    if not eligible:
        raise RuntimeError('No fresh idle GPU; wait without foreign intervention')
    wanted = eligible[0]
    chosen = [row.split(',') for row in rows if row.split(',')[0].strip() == wanted]
    if len(chosen) != 1:
        raise RuntimeError('Exactly one currently visible chosen GPU UUID')
    used, utilization = (int(value.strip()) for value in chosen[0][1:])
    if wanted in occupied or used < 0 or utilization < 0 or used > cfg['GPU_idle_max_used_MiB'] or utilization > cfg['GPU_idle_max_utilization_percent']:
        raise RuntimeError('Chosen GPU no longer idle; do not intervene with external work')
    return {'GPU_uuid': wanted, 'memory_used_MiB': used, 'utilization_percent': utilization,
            'chosen_GPU_compute_applications_before_launch': 0}


def worker(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(out / '_fresh_no_write_cache'):
        raise ValueError('Fresh no-write worker required')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy,
                                allow_public_api=True)
    for path in SOURCES:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Registered source changed')
    frozen = json.loads((ROOT / cfg['frozen_protocol']).read_text())
    approval = json.loads((ROOT / cfg['approval']).read_text())
    if (frozen['status'] != 'FROZEN_DATA_ROLES_AND_GT_CONTROL_CACHE_IO_NOT_FINAL_METHOD'
            or approval['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or approval['bound_private_assignment_sha256'] != frozen['input_sha256']['private_global_assignment']
            or approval['new_nonKnown_semantic_GT_or_Novel_vocabulary_authorized'] is not False
            or cfg['role'] != 'independent_final_pseudo'
            or cfg.get('independent_final_pseudo_cache_only') is not True
            or cfg['pilot_videos'] != 2
            or cfg['maximum_observations_per_track'] is not None
            or cfg['full_cache_or_optimizer_automatic_start'] is not False):
        raise ValueError('Exact independent-final-pseudo bounded cache pilot required')
    inputs = {'private_assignment': 'private_global_assignment', 'private_image_plan': 'private_image_plan',
              'private_physical_GT': 'private_physical_GT'}
    for name, field in inputs.items():
        path = ROOT / cfg[name]
        if path.is_symlink() or sha256_file(path) != frozen['input_sha256'][field]:
            raise ValueError('Hash-bound private index changed')
    if out.exists():
        raise ValueError('Never overwrite previous independent pilot')
    out.mkdir(parents=True)
    started, peak = time.monotonic(), 0
    prior_bytes = sum(path.stat().st_size for path in (ROOT / 'outputs/trackocd_cvpr').rglob('*')
                      if path.is_file() and not path.is_symlink())

    def guard():
        nonlocal peak
        peak = max(peak, process_memory()['guarded_process_RSS_bytes'])
        reason = resources_ok(cfg, safety, ROOT, started, peak)
        output_bytes = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
        if reason or output_bytes > cfg['output_bytes_ceiling'] or prior_bytes + output_bytes > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned independent pilot resource guard:' + str(reason))

    atomic_json(out / 'launch.json', {
        'preregistration_commit': args.preregistration_commit,
        'live_remote_registration_proof': proof,
        'source_sha256': {path: sha256_file(ROOT / path) for path in SOURCES},
        'frozen_protocol_sha256': sha256_file(ROOT / cfg['frozen_protocol']),
        'scope': 'Independent-final-pseudo Train GT-control cache pilot; no semantic labels/model selection/optimizer'})
    try:
        import numpy as np
        plan = json.loads((ROOT / cfg['private_image_plan']).read_text())
        if plan.get('semantic_labels_or_GT_boxes_in_image_plan') is not False:
            raise ValueError('Label-free image plan required')
        final_ids = assigned_video_ids(ROOT / cfg['private_assignment'], 'independent_final_pseudo', expected_count=68)
        selected = sorted(final_ids)[:cfg['pilot_videos']]
        tasks = selected_video_plans(ROOT / cfg['private_physical_GT'], selected, plan)
        if (sum(len(arrays['track_id']) for _, _, arrays in tasks) != cfg['expected_tracks']
                or sum(len(arrays['visual']) for _, _, arrays in tasks) != cfg['expected_observations']
                or sum(len(images) for images, _, _ in tasks) != cfg['expected_metadata_frames']):
            raise ValueError('Exact independent-final pilot denominators required')
        common = json.loads((ROOT / cfg['common_config']).read_text())
        repo = ROOT / common['local_repository']
        checkpoint = ROOT / common['checkpoint']
        if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode().strip() != common['upstream_commit']
                or subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip()
                or sha256_file(checkpoint) != common['checkpoint_sha256']):
            raise ValueError('Clean pinned encoder / checkpoint required')
        source_hashes, absent, tree_count = verified_encoder_sources(repo, common['upstream_commit'])
        guard(); idle = require_idle_gpu(cfg)
        os.environ['CUDA_VISIBLE_DEVICES'] = idle['GPU_uuid']
        import torch
        from PIL import Image
        from torchvision import transforms
        from scripts.trackocd_v2.build_common_features import crop_box
        sys.path.insert(0, str(repo))
        from dinov2.hub.backbones import dinov2_vitb14
        torch.set_num_threads(1); torch.manual_seed(1027)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        frame_root = Path(cfg['frame_root']).resolve(); train_root = (frame_root / 'train').resolve()
        allowed = {(frame_root / row['file_name']).resolve() for images, _, _ in tasks for row in images}
        if any(not path.is_relative_to(train_root) for path in allowed):
            raise ValueError('Only selected Train pixel paths')

        def barrier(event, values):
            if event == 'socket.connect':
                raise PermissionError('No network after source registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                path = Path(os.fsdecode(values[0])).resolve()
                if (path.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or path.is_relative_to(frame_root)) and path not in allowed:
                    raise PermissionError('Only selected Train pixels; no annotations or protected opens')

        sys.addaudithook(barrier)
        model = dinov2_vitb14(pretrained=False)
        imported = 0
        for name, module in list(sys.modules.items()):
            if name != 'dinov2' and not name.startswith('dinov2.'):
                continue
            actual = getattr(module, '__file__', None)
            if actual is None:
                continue
            path = Path(actual).resolve()
            if not path.is_relative_to(repo.resolve()):
                raise ValueError('Encoder module outside pinned source')
            relative = path.relative_to(repo.resolve()).as_posix()
            if relative not in source_hashes or sha256_file(path) != source_hashes[relative]:
                raise ValueError('Imported encoder bytes changed')
            imported += 1
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        model.load_state_dict(state, strict=True); del state
        model.eval().requires_grad_(False); before = digest(model)
        idle = require_idle_gpu(cfg)
        GPU_started = time.monotonic()
        torch.cuda.set_per_process_memory_fraction(cfg['GPU_peak_bytes'] / torch.cuda.get_device_properties(0).total_memory)
        model.to('cuda:0'); torch.cuda.reset_peak_memory_stats()
        transform = transforms.Compose([transforms.Resize(tuple(common['resize']), interpolation=Image.Resampling.BILINEAR),
            transforms.ToTensor(), transforms.Normalize(common['normalization_mean'], common['normalization_std'])])
        first_tensor = first_visual = None
        shards = []; private_index = []; pixel_receipts = []; forwards = decoded = 0

        def encode(batch):
            guard()
            if torch.cuda.max_memory_reserved() > cfg['GPU_peak_bytes'] or torch.cuda.mem_get_info()[0] < cfg['GPU_free_reserve_bytes']:
                raise RuntimeError('Owned GPU ceiling / free reserve')
            with torch.inference_mode():
                value = torch.nn.functional.normalize(model.forward_features(batch.to('cuda:0'))['x_norm_clstoken'], dim=-1)
            torch.cuda.synchronize()
            array = value.cpu().numpy()
            if not np.isfinite(array).all() or array.shape != (len(batch), 768):
                raise ValueError('Finite frozen common768')
            return array

        for ordinal, (images, boxes, arrays) in enumerate(tasks):
            for row in images:
                positions = np.flatnonzero(arrays['image_id'] == row['id'])
                if len(positions) == 0:
                    continue
                path = (frame_root / row['file_name']).resolve(); stat = path.stat()
                pixel_receipts.append({'sha256': sha256_file(path), 'bytes': stat.st_size})
                with Image.open(path) as raw:
                    rgb = raw.convert('RGB'); rgb.load()
                    if rgb.size != (row['width'], row['height']):
                        raise ValueError('Decoded current geometry dimensions differ')
                    for start in range(0, len(positions), cfg['batch_size']):
                        selected_positions = positions[start:start + cfg['batch_size']]
                        batch = torch.stack([transform(crop_box(rgb, boxes[position], common['crop_context_fraction_per_side']))
                                             for position in selected_positions])
                        value = encode(batch); forwards += 1
                        arrays['visual'][selected_positions] = value.astype(np.float16)
                        if first_tensor is None:
                            first_tensor, first_visual = batch[:1].clone(), value[:1].copy()
                if (path.stat().st_size, path.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise ValueError('Current pixels changed')
                decoded += 1
                atomic_json(out / 'progress.json', {'completed_videos': ordinal,
                    'decoded_independent_final_images': decoded, 'forward_batches': forwards,
                    'not_optimizer_or_fullscope': True})
            check = check_every_prefix(arrays); guard()
            record = write_atomic_shard(out, ordinal, arrays, {
                'registered_commit': args.preregistration_commit,
                'frozen_protocol_sha256': sha256_file(ROOT / cfg['frozen_protocol']),
                'physical_index_sha256': frozen['input_sha256']['private_physical_GT'],
                'image_plan_sha256': frozen['input_sha256']['private_image_plan'],
                'checkpoint_sha256': common['checkpoint_sha256'], 'opaque_video_id': selected[ordinal]})
            reopened, _ = read_verified_shard(out, ordinal)
            for name, values in arrays.items():
                np.testing.assert_array_equal(values, reopened[name])
            shards.append({'ordinal': ordinal, 'tracks': record['tracks'], 'observations': record['observations'],
                           'metadata_frames': record['metadata_frames'], 'empty_GT_frames': record['empty_GT_frames'],
                           'private_payload_bytes': record['payload']['bytes'], 'private_payload_sha256': record['payload']['sha256'],
                           'every_visible_prefix': check, 'reopen_exact': True})
            private_index.append({'ordinal': ordinal, 'source_directory': cfg['output_directory'],
                                  'opaque_video_id': selected[ordinal], 'payload_sha256': record['payload']['sha256']})
            atomic_json(out / 'private_shard_index.json', private_index)
            atomic_json(out / 'progress.json', {'completed_videos': ordinal + 1,
                'decoded_independent_final_images': decoded, 'forward_batches': forwards,
                'not_optimizer_or_fullscope': True})
        repeat = encode(first_tensor); forwards += 1
        GPU_peak = torch.cuda.max_memory_reserved()
        model.to('cpu'); torch.cuda.empty_cache(); guard()
        with torch.inference_mode():
            CPU = torch.nn.functional.normalize(model.forward_features(first_tensor)['x_norm_clstoken'], dim=-1).numpy()

        def comparison(reference, value):
            error = float(np.max(np.abs(reference - value)))
            cosine = float(np.sum(reference * value) / (np.linalg.norm(reference) * np.linalg.norm(value)))
            if error > cfg['precision_max_absolute_error'] or cosine < cfg['precision_min_cosine']:
                raise ValueError('Preregistered CPU/GPU or batch-repeat precision gate')
            return {'max_absolute_error': error, 'cosine': cosine}

        GPU_repeat = comparison(first_visual, repeat); CPU_GPU = comparison(first_visual, CPU)
        stored = first_visual.astype(np.float16).astype(np.float32); stored /= np.linalg.norm(stored, axis=1, keepdims=True)
        fp16 = comparison(first_visual, stored)
        if digest(model) != before or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError('Frozen model parameters changed')
        guard(); atomic_json(out / 'private_pixel_receipts.json', pixel_receipts)
        result = {
            'status': 'VERIFIED_INDEPENDENT_FINAL_PSEUDO_GT_CONTROL_CACHE_PILOT_NOT_PRED_OR_TRAINED',
            'live_remote_registration_proof': proof, 'shards': shards,
            'pilot_videos': len(tasks), 'tracks': sum(record['tracks'] for record in shards),
            'videos': len(tasks), 'observations': sum(record['observations'] for record in shards),
            'metadata_frames': sum(record['metadata_frames'] for record in shards),
            'decoded_independent_final_images': decoded, 'GPU_forward_batches_including_repeat': forwards,
            'CPU_comparison_forwards': 1, 'GPU_idle_before_owned_allocation': idle,
            'frozen_parameters_unchanged': True, 'frozen_parameter_sha256': before,
            'encoder_source_files_verified': len(source_hashes), 'encoder_source_tree_entries': tree_count,
            'sparse_absent_not_imported': absent, 'imported_encoder_modules_verified': imported,
            'precision_GPU_batch_single_repeat': GPU_repeat, 'precision_CPU_GPU_firstcrop': CPU_GPU,
            'precision_float16_reader_firstcrop': fp16,
            'semantic_GT_roles_IDs_futurelength_in_numeric_forward': False,
            'development_final_or_protected_pixels_read': False, 'development_pixels_read': False,
            'independent_final_or_protected_pixels_read': False, 'independent_final_pseudo_pixels_read': True,
            'development_or_final_semantic_values_decoded': False,
            'optimizer_or_formal_training_started': False, 'full407_cache_automatic_launch': False,
            'qualified_Pred_or_Goal_A_B_C_achieved': False,
            'resources': {'wall_seconds_excluding_network': time.monotonic() - started,
                'peak_guarded_RSS_bytes': peak, 'GPU_peak_reserved_bytes': GPU_peak,
                'frozen_cache_GPU_hours': (time.monotonic() - GPU_started) / 3600,
                'training_GPU_hours': 0, 'prior_CVPR_output_bytes': prior_bytes}}
        guard(); atomic_json(out / 'result.json', result); print(json.dumps(result), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False})
        raise


def supervise(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    receipt = out.with_name(out.name + f'_supervisor_attempt_{args.attempt}.json')
    if out.exists() or receipt.exists():
        raise ValueError('Preserve every previous owned-worker attempt')
    if args.attempt != 1:
        raise ValueError('This first pilot has no automatic retry path')
    started, peak, reason, peak_threads = time.monotonic(), 0, None, 0
    if resources_ok(cfg, safety, ROOT, started, 0):
        raise RuntimeError('Pre-spawn resource guard')
    child = subprocess.Popen([sys.executable, '-B', '-X', 'pycache_prefix=' + str(out / '_fresh_no_write_cache'),
                              str(ROOT / 'scripts/trackocd_cvpr/independent_final_GT_video_cache.py'), '--worker',
                              '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy,
                              '--attempt', str(args.attempt), '--config', args.config], cwd=ROOT)
    try:
        while child.poll() is None:
            status_path = Path(f'/proc/{child.pid}/status'); rollup_path = Path(f'/proc/{child.pid}/smaps_rollup')
            try:
                status = status_path.read_text(); measured = process_memory(status, rollup_path.read_text())
                peak = max(peak, measured['guarded_process_RSS_bytes'])
                fields = {line.split(':', 1)[0]: line.split(':', 1)[1] for line in status.splitlines() if ':' in line}
                peak_threads = max(peak_threads, int(fields['Threads'].strip()))
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                if child.poll() is not None:
                    break
                if time.monotonic() - started > 2:
                    reason = 'owned_memory_counter_unavailable'; child.kill(); break
                time.sleep(cfg['supervisor_poll_seconds']); continue
            reason = resources_ok(cfg, safety, ROOT, started, peak)
            output_bytes = sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) if out.exists() else 0
            if output_bytes > cfg['output_bytes_ceiling']:
                reason = 'owned_output_size'
            if reason:
                if child.poll() is None:
                    child.kill()
                break
            time.sleep(cfg['supervisor_poll_seconds'])
        code = child.wait(timeout=10)
    except BaseException:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10); raise
    atomic_json(receipt, {'attempt': args.attempt, 'owned_worker_exit_code': code,
        'guard_stop_reason': reason, 'peak_sampled_owned_worker_RSS_bytes': peak,
        'peak_sampled_worker_threads': peak_threads,
        'wall_seconds_including_imports_network_and_solver': time.monotonic() - started,
        'only_new_owned_Popen_worker_managed': True, 'training_GPU_hours': 0,
        'successful_owned_execution': code == 0 and reason is None})
    return 0 if code == 0 and reason is None else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True); parser.add_argument('--git-proxy', default='')
    parser.add_argument('--config', choices=(CONFIG,), default=CONFIG)
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--attempt', type=int, choices=(1,), default=1)
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker:
        worker(args, cfg, safety); return 0
    return supervise(args, cfg, safety)


if __name__ == '__main__':
    sys.exit(main())
