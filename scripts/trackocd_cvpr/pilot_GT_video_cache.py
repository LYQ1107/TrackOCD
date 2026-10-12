#!/usr/bin/env python3
"""Two fit-video frozen GT-control shards; no optimizer or whole-scope launch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration
from scripts.trackocd_cvpr.audit_joint_known_protocol import supervise, resources_ok
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory
from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import fit_video_ids, digest, verified_encoder_sources

CONFIG = 'configs/trackocd_cvpr/pilot_GT_video_cache_v1.json'
FIT_CONFIG = 'configs/trackocd_cvpr/fit_GT_video_cache_v1.json'
POLICY_DEV_CONFIG = 'configs/trackocd_cvpr/policy_dev_GT_video_cache_v1.json'
SOURCES = (CONFIG, 'scripts/trackocd_cvpr/pilot_GT_video_cache.py',
    'src/trackocd_cvpr/GT_video_cache.py', 'scripts/trackocd_cvpr/audit_joint_known_protocol.py',
    'scripts/trackocd_cvpr/smoke_selected_frozen_encoder.py',
    'scripts/trackocd_cvpr/audit_train_metadata_containment.py', 'src/trackocd_cvpr/remote_registration.py',
    'src/trackocd_cvpr/scoped_train_census.py', 'src/trackocd_cvpr/metadata_only.py',
    'src/trackocd_v2/io.py', 'src/trackocd_core/features.py',
    'scripts/trackocd_v2/build_common_features.py', 'configs/trackocd_core/common_features.json',
    'configs/trackocd_cvpr/formal_research_scope_approval_v1.json',
    'configs/trackocd_cvpr/frozen_global_data_GT_control_v1.json',
    'configs/trackocd_cvpr/autonomous_research.json')


def require_idle_gpu(cfg):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], timeout=10).decode().splitlines()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                   '--format=csv,noheader'], timeout=10).decode().splitlines()
    occupied = {r.strip() for r in apps}
    wanted = cfg['GPU_uuid']
    if wanted == 'FIRST_FRESH_IDLE_UUID':
        eligible = []
        for row in rows:
            fields = [v.strip() for v in row.split(',')]
            if len(fields) != 3 or fields[0] in occupied:
                continue
            try: used, utilization = map(int, fields[1:])
            except ValueError: continue  # unknown/N/A is not idle
            if 0 <= used <= cfg['GPU_idle_max_used_MiB'] and 0 <= utilization <= cfg['GPU_idle_max_utilization_percent']:
                eligible.append(fields[0])
        if not eligible:
            raise RuntimeError('No fresh idle GPU; wait without foreign intervention')
        wanted = eligible[0]
    chosen = [r.split(',') for r in rows if r.split(',')[0].strip() == wanted]
    if len(chosen) != 1:
        raise RuntimeError('Exactly one currently visible chosen GPU UUID')
    used, utilization = (int(v.strip()) for v in chosen[0][1:])
    if (wanted in occupied or used < 0 or utilization < 0 or used > cfg['GPU_idle_max_used_MiB']
            or utilization > cfg['GPU_idle_max_utilization_percent']):
        raise RuntimeError('Chosen GPU no longer idle; do not intervene with external work')
    return {'GPU_uuid': wanted, 'memory_used_MiB': used, 'utilization_percent': utilization,
            'chosen_GPU_compute_applications_before_launch': 0}


def worker(args, cfg, safety):
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(ROOT / cfg['output_directory'] / '_fresh_no_write_cache'):
        raise ValueError('Fresh no-write worker required')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for p in SOURCES:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + p], cwd=ROOT) != (ROOT / p).read_bytes():
            raise ValueError('Registered source changed')
    frozen = json.loads((ROOT / cfg['frozen_protocol']).read_text())
    approval = json.loads((ROOT / cfg['approval']).read_text())
    bounded_fit = cfg.get('bounded_fit_partition_only', False)
    policy_dev = cfg.get('policy_dev_cache_only', False)
    if (frozen['status'] != 'FROZEN_DATA_ROLES_AND_GT_CONTROL_CACHE_IO_NOT_FINAL_METHOD'
            or approval['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or approval['bound_private_assignment_sha256'] != frozen['input_sha256']['private_global_assignment']
            or approval['new_nonKnown_semantic_GT_or_Novel_vocabulary_authorized'] is not False
            or (bounded_fit and policy_dev)
            or cfg['pilot_videos'] != (82 if policy_dev else 257 if bounded_fit else 2)
            or cfg['maximum_observations_per_track'] is not None
            or cfg['full_cache_or_optimizer_automatic_start'] is not False):
        raise ValueError('Exact authorized frozen tiny pilot only')
    names = {'private_assignment': 'private_global_assignment', 'private_image_plan': 'private_image_plan',
             'private_physical_GT': 'private_physical_GT'}
    for name, field in names.items():
        path = ROOT / cfg[name]
        if path.is_symlink() or sha256_file(path) != frozen['input_sha256'][field]:
            raise ValueError('Hash-bound private index changed')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Never overwrite previous attempt')
    out.mkdir(parents=True)
    started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*')
                      if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, process_memory()['guarded_process_RSS_bytes'])
        reason = resources_ok(cfg, safety, ROOT, started, peak)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes + size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned pilot resource guard:' + str(reason))
    atomic_json(out / 'launch.json', {'preregistration_commit': args.preregistration_commit,
        'live_remote_registration_proof': proof, 'source_sha256': {p: sha256_file(ROOT / p) for p in SOURCES},
        'frozen_protocol_sha256': sha256_file(ROOT / cfg['frozen_protocol']),
        'scope': ('Fixed policy/dev-only GT-control preparation; dev pixels authorized, no dev labels/final/optimizer'
                  if policy_dev else 'Fixed fit-only uncapped annotated GT-control shards; bounded preparation, not Pred or optimizer')})
    try:
        import numpy as np
        from src.trackocd_cvpr.GT_video_cache import (selected_video_plans, policy_dev_video_ids, write_atomic_shard, read_verified_shard,
                                                    check_every_prefix, validate_reuse_plan)
        video_roles = policy_dev_video_ids(ROOT / cfg['private_assignment']) if policy_dev else {}
        fit = sorted(video_roles) if policy_dev else sorted(fit_video_ids(ROOT / cfg['private_assignment']))[:cfg['pilot_videos']]
        plan = json.loads((ROOT / cfg['private_image_plan']).read_text())
        tasks = selected_video_plans(ROOT / cfg['private_physical_GT'], fit, plan)
        if (sum(len(a['track_id']) for _, _, a in tasks) != cfg['expected_tracks']
                or sum(len(a['visual']) for _, _, a in tasks) != cfg['expected_observations']
                or sum(len(images) for images, _, _ in tasks) != cfg['expected_metadata_frames']):
            raise ValueError('Uncapped exact preregistered pilot denominators')
        common = json.loads((ROOT / cfg['common_config']).read_text())
        reused = {}
        if bounded_fit or policy_dev:
            pilot = ROOT / cfg['qualified_pilot_directory']
            if (cfg['qualified_pilot_directory'] != 'outputs/trackocd_cvpr/cvpr2/pilot_GT_video_cache_v1'
                    or sha256_file(pilot / 'result.json') != cfg['qualified_pilot_result_sha256']
                    or cfg['reuse_first_video_shards'] != (0 if policy_dev else 2)):
                raise ValueError('Exact prior qualified two-video pilot')
            terminal = json.loads(pilot.with_name(pilot.name + '_supervisor_attempt_1.json').read_text())
            if not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0 or terminal['guard_stop_reason'] is not None:
                raise ValueError('Prior payload completion not sufficient without overall pilot qualification')
            for ordinal in range(cfg['reuse_first_video_shards']):
                arrays, record = read_verified_shard(pilot, ordinal)
                validate_reuse_plan(arrays, tasks[ordinal][2], record, video=fit[ordinal],
                    frozen_sha=sha256_file(ROOT / cfg['frozen_protocol']),
                    physical_sha=frozen['input_sha256']['private_physical_GT'], images_sha=frozen['input_sha256']['private_image_plan'],
                    checkpoint_sha=common['checkpoint_sha256'])
                reused[ordinal] = (arrays, record)
        repo = ROOT / common['local_repository']; checkpoint = ROOT / common['checkpoint']
        if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode().strip() != common['upstream_commit']
                or subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip()
                or sha256_file(checkpoint) != common['checkpoint_sha256']):
            raise ValueError('Clean pinned encoder / checkpoint')
        source_hashes, absent, tree_count = verified_encoder_sources(repo, common['upstream_commit'])
        guard(); idle = require_idle_gpu(cfg)
        cfg = {**cfg, 'GPU_uuid': idle['GPU_uuid']}  # runtime resource binding, not experiment/data selection
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['GPU_uuid']
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
        allowed = {(frame_root / r['file_name']).resolve() for images, _, _ in tasks for r in images}
        if any(not p.is_relative_to(train_root) for p in allowed):
            raise ValueError('Only selected Train pixel paths')
        def barrier(event, values):
            if event == 'socket.connect':
                raise PermissionError('No network after source registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                p = Path(os.fsdecode(values[0])).resolve()
                if (p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or p.is_relative_to(frame_root)) and p not in allowed:
                    raise PermissionError('Only selected fit pixels; no original annotations or protected opens')
        sys.addaudithook(barrier)
        model = dinov2_vitb14(pretrained=False)
        imported = 0
        for name, module in list(sys.modules.items()):
            if name == 'dinov2' or name.startswith('dinov2.'):
                actual = getattr(module, '__file__', None)
                if actual is None: continue
                p = Path(actual).resolve()
                if not p.is_relative_to(repo.resolve()):
                    raise ValueError('Encoder module outside pinned source')
                rel = p.relative_to(repo.resolve()).as_posix()
                if rel not in source_hashes or sha256_file(p) != source_hashes[rel]:
                    raise ValueError('Imported encoder bytes changed')
                imported += 1
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        model.load_state_dict(state, strict=True); del state
        model.eval().requires_grad_(False); before = digest(model)
        idle = require_idle_gpu(cfg)  # recheck immediately before owned CUDA allocation
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
        for ordinal, (images, boxes, a) in enumerate(tasks):
            if ordinal in reused:
                a, record = reused[ordinal]
                check = check_every_prefix(a); guard()
                shards.append({'ordinal': ordinal, 'tracks': record['tracks'], 'observations': record['observations'],
                    'metadata_frames': record['metadata_frames'], 'empty_GT_frames': record['empty_GT_frames'],
                    'private_payload_bytes': record['payload']['bytes'], 'private_payload_sha256': record['payload']['sha256'],
                    'every_visible_prefix': check, 'reopen_exact': True, 'reused_without_extraction': True})
                private_index.append({'ordinal': ordinal, 'source_directory': cfg['qualified_pilot_directory'],
                                      'video_id': fit[ordinal], 'payload_sha256': record['payload']['sha256']})
                atomic_json(out / 'progress.json', {'completed_videos': ordinal + 1, 'decoded_selected_images' if policy_dev else 'decoded_fit_images': decoded,
                    'forward_batches': forwards, 'reused_video_shards': len(reused), 'not_optimizer_or_fullscope': True})
                continue
            for row in images:
                positions = np.flatnonzero(a['image_id'] == row['id'])
                if len(positions) == 0:
                    continue  # frame remains in clock, not a BG sample
                path = (frame_root / row['file_name']).resolve(); stat = path.stat()
                pixel_receipts.append({'sha256': sha256_file(path), 'bytes': stat.st_size})
                with Image.open(path) as raw:
                    rgb = raw.convert('RGB'); rgb.load()
                    if rgb.size != (row['width'], row['height']):
                        raise ValueError('Decoded current geometry dimensions differ')
                    for start in range(0, len(positions), cfg['batch_size']):
                        selected = positions[start:start + cfg['batch_size']]
                        batch = torch.stack([transform(crop_box(rgb, boxes[o], common['crop_context_fraction_per_side'])) for o in selected])
                        value = encode(batch); forwards += 1
                        a['visual'][selected] = value.astype(np.float16)
                        if first_tensor is None:
                            first_tensor, first_visual = batch[:1].clone(), value[:1].copy()
                if (path.stat().st_size, path.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise ValueError('Current pixels changed')
                decoded += 1
                atomic_json(out / 'progress.json', {'completed_videos': ordinal, 'decoded_selected_images' if policy_dev else 'decoded_fit_images': decoded,
                    'forward_batches': forwards, 'not_optimizer_or_fullscope': True})
            check = check_every_prefix(a); guard()
            record = write_atomic_shard(out, ordinal, a, {'registered_commit': args.preregistration_commit,
                'frozen_protocol_sha256': sha256_file(ROOT / cfg['frozen_protocol']),
                'physical_index_sha256': frozen['input_sha256']['private_physical_GT'],
                'image_plan_sha256': frozen['input_sha256']['private_image_plan'],
                'checkpoint_sha256': common['checkpoint_sha256'], 'opaque_video_id': fit[ordinal]})
            reopened, _ = read_verified_shard(out, ordinal)
            for name, values in a.items(): np.testing.assert_array_equal(values, reopened[name])
            shards.append({'ordinal': ordinal, 'tracks': record['tracks'], 'observations': record['observations'],
                           'metadata_frames': record['metadata_frames'], 'empty_GT_frames': record['empty_GT_frames'],
                           'private_payload_bytes': record['payload']['bytes'], 'private_payload_sha256': record['payload']['sha256'],
                           'every_visible_prefix': check, 'reopen_exact': True})
            private_index.append({'ordinal': ordinal, 'source_directory': cfg['output_directory'],
                                  'video_id': fit[ordinal], 'payload_sha256': record['payload']['sha256']})
            atomic_json(out / 'private_shard_index.json', private_index)
            atomic_json(out / 'progress.json', {'completed_videos': ordinal + 1, 'decoded_selected_images' if policy_dev else 'decoded_fit_images': decoded,
                'forward_batches': forwards, 'reused_video_shards': len(reused), 'not_optimizer_or_fullscope': True})
        repeat = encode(first_tensor); forwards += 1
        GPU_peak = torch.cuda.max_memory_reserved()
        model.to('cpu'); torch.cuda.empty_cache()
        guard()
        with torch.inference_mode():
            CPU = torch.nn.functional.normalize(model.forward_features(first_tensor)['x_norm_clstoken'], dim=-1).numpy()
        def comparison(reference, value):
            error = float(np.max(np.abs(reference - value)))
            cosine = float(np.sum(reference * value) / (np.linalg.norm(reference) * np.linalg.norm(value)))
            if error > cfg['precision_max_absolute_error'] or cosine < cfg['precision_min_cosine']:
                raise ValueError('Preregistered CPU/GPU or batch-repeat precision gate')
            return {'max_absolute_error': error, 'cosine': cosine}
        GPU_repeat = comparison(first_visual, repeat); CPU_GPU = comparison(first_visual, CPU)
        stored = first_visual.astype(np.float16).astype(np.float32)
        stored /= np.linalg.norm(stored, axis=1, keepdims=True)
        fp16 = comparison(first_visual, stored)
        if digest(model) != before or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Frozen model parameters changed')
        guard(); atomic_json(out / 'private_pixel_receipts.json', pixel_receipts)
        role_totals = {r: {'videos': 0, 'tracks': 0, 'observations': 0, 'metadata_frames': 0, 'empty_GT_frames': 0}
                       for r in set(video_roles.values())}
        for video, shard in zip(fit, shards):
            if policy_dev:
                totals = role_totals[video_roles[video]]; totals['videos'] += 1
                for name in ('tracks', 'observations', 'metadata_frames', 'empty_GT_frames'): totals[name] += shard[name]
        if policy_dev:
            for role, expected in cfg['expected_role_totals'].items():
                if any(role_totals[role][k] != v for k, v in expected.items()):
                    raise ValueError('Every policy/dev physical denominator must reconcile separately')
        result = {'status': ('VERIFIED_POLICY_DEV_UNCAPPED_GT_CONTROL_CACHE_NOT_FINAL_OR_PRED_OR_TRAINED' if policy_dev else
                            'VERIFIED_FIT_PARTITION_UNCAPPED_GT_CONTROL_FEATURE_CACHE_NOT_PRED_OR_TRAINED' if bounded_fit else
                             'VERIFIED_TWO_FIT_VIDEO_UNCAPPED_GT_CONTROL_CACHE_PILOT_NOT_PRED_OR_TRAINED'),
            'live_remote_registration_proof': proof, 'shards': shards,
            'pilot_videos': len(tasks), 'tracks': sum(r['tracks'] for r in shards),
            'videos': len(tasks), 'reused_video_shards_without_extraction': len(reused),
            'observations': sum(r['observations'] for r in shards), 'metadata_frames': sum(r['metadata_frames'] for r in shards),
            'decoded_selected_images' if policy_dev else 'decoded_fit_images': decoded,
            'GPU_forward_batches_including_repeat': forwards, 'CPU_comparison_forwards': 1,
            'GPU_idle_before_owned_allocation': idle, 'frozen_parameters_unchanged': True, 'frozen_parameter_sha256': before,
            'encoder_source_files_verified': len(source_hashes), 'encoder_source_tree_entries': tree_count,
            'sparse_absent_not_imported': absent, 'imported_encoder_modules_verified': imported,
            'precision_GPU_batch_single_repeat': GPU_repeat, 'precision_CPU_GPU_firstcrop': CPU_GPU,
            'precision_float16_reader_firstcrop': fp16, 'semantic_GT_roles_IDs_futurelength_in_numeric_forward': False,
            'development_final_or_protected_pixels_read': policy_dev,
            'development_pixels_read': policy_dev, 'independent_final_or_protected_pixels_read': False,
            'development_or_final_semantic_values_decoded': False, 'role_totals': role_totals,
            'optimizer_or_formal_training_started': False,
            'full407_cache_automatic_launch': False, 'qualified_Pred_or_Goal_A_B_C_achieved': False,
            'resources': {'wall_seconds_excluding_network': time.monotonic() - started, 'peak_guarded_RSS_bytes': peak,
                'GPU_peak_reserved_bytes': GPU_peak, 'frozen_cache_GPU_hours': (time.monotonic() - GPU_started) / 3600,
                'GPU_hours_measurement': 'First owned CUDA initialization through result, including CPU comparison while context remains; terminal supervisor wall is also recorded',
                'training_GPU_hours': 0,
                'prior_CVPR_output_bytes': prior_bytes}}
        guard(); atomic_json(out / 'result.json', result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False})
        raise


def main():
    global SOURCES
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True); parser.add_argument('--git-proxy', default='')
    parser.add_argument('--config', choices=(CONFIG, FIT_CONFIG, POLICY_DEV_CONFIG), default=CONFIG)
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--attempt', type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    SOURCES = (args.config, *SOURCES[1:])
    if args.config in (FIT_CONFIG, POLICY_DEV_CONFIG):
        SOURCES = (*SOURCES[:-1], 'outputs/trackocd_cvpr/cvpr2/pilot_GT_video_cache_v1/result.json',
                   'outputs/trackocd_cvpr/cvpr2/pilot_GT_video_cache_v1_supervisor_attempt_1.json', SOURCES[-1])
    cfg = json.loads((ROOT / args.config).read_text())
    safety = json.loads((ROOT / SOURCES[-1]).read_text())
    if args.worker: worker(args, cfg, safety); return 0
    return supervise(args, cfg, safety)


if __name__ == '__main__':
    sys.exit(main())
