#!/usr/bin/env python3
"""Two fit-only whole-image CPU DINO descriptors; not crop/track training."""
import argparse
import hashlib
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
from src.trackocd_cvpr.scoped_train_census import top_ranges, object_ranges, array_ranges, integer_value
from src.trackocd_cvpr.remote_registration import verify_registration
from scripts.trackocd_cvpr.audit_joint_known_protocol import supervise, resources_ok
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory

SOURCES = ('configs/trackocd_cvpr/selected_frozen_encoder_smoke_v1.json',
           'scripts/trackocd_cvpr/smoke_selected_frozen_encoder.py',
           'scripts/trackocd_cvpr/audit_joint_known_protocol.py',
           'scripts/trackocd_cvpr/audit_train_metadata_containment.py',
           'src/trackocd_cvpr/remote_registration.py', 'src/trackocd_cvpr/scoped_train_census.py',
           'src/trackocd_cvpr/metadata_only.py', 'src/trackocd_v2/io.py',
           'configs/trackocd_core/common_features.json', 'scripts/trackocd_v2/build_common_features.py',
           'configs/trackocd_cvpr/autonomous_research.json')


def fit_video_ids(path):
    # No classes, role_class_support or other video lists are JSON-decoded.
    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        fields = top_ranges(data)
        assignment = object_ranges(data, *fields['assignment'])
        roles = object_ranges(data, *assignment['video_ids'])
        ids = [integer_value(data, span) for span in array_ranges(data, *roles['fit_known'])]
    if len(ids) != 257 or len(set(ids)) != len(ids):
        raise ValueError('Exact fixed fit-only video scope')
    return set(ids)


def digest(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        value.update(name.encode()); value.update(str(array.dtype).encode())
        value.update(str(array.shape).encode()); value.update(array.tobytes())
    return value.hexdigest()


def verified_encoder_sources(repo, commit, run=subprocess.check_output):
    source_hashes, sparse_absent = {}, []
    records = run(['git', 'ls-files', '-t', '*.py'], cwd=repo).decode().splitlines()
    for record in records:
        tag, path = record.split(' ', 1)
        target = repo / path
        if not target.is_file():
            if tag != 'S':
                raise ValueError('Missing encoder source not marked skip-worktree')
            sparse_absent.append(path)
            continue
        original = run(['git', 'show', commit + ':' + path], cwd=repo)
        if target.read_bytes() != original:
            raise ValueError('Encoder tracked source bytes differ')
        source_hashes[path] = hashlib.sha256(original).hexdigest()
    if not source_hashes:
        raise ValueError('No verified encoder source')
    return source_hashes, sparse_absent, len(records)


def worker(args, cfg, safety):
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(ROOT / cfg['output_directory'] / '_fresh_no_write_cache'):
        raise ValueError('Fresh no-write bytecode worker required')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for path in SOURCES:
        if subprocess.check_output(['git', 'show', f'{args.preregistration_commit}:{path}'], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Registered source changed')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve previous output')
    out.mkdir(parents=True)
    started, peak = time.monotonic(), 0
    def guard():
        nonlocal peak
        peak = max(peak, process_memory()['guarded_process_RSS_bytes'])
        reason = resources_ok(cfg, safety, ROOT, started, peak)
        if reason or sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > cfg['output_bytes_ceiling']:
            raise RuntimeError('Owned CPU smoke guard:' + str(reason))
    atomic_json(out / 'launch.json', {'preregistration_commit': args.preregistration_commit,
        'live_remote_registration_proof': proof, 'source_sha256': {p: sha256_file(ROOT / p) for p in SOURCES},
        'scope': 'Only two fit images, CPU, no GT/class/role/text in numeric forward; not trajectories/crop validity'})
    try:
        common = json.loads((ROOT / cfg['common_config']).read_text())
        inputs = {'private_assignment': ROOT / cfg['private_assignment'], 'private_image_plan': ROOT / cfg['private_image_plan']}
        for name, path in inputs.items():
            if path.is_symlink() or sha256_file(path) != cfg[name + '_sha256']:
                raise ValueError('Fixed scalar metadata input hash')
        fit = fit_video_ids(inputs['private_assignment'])
        plan = json.loads(inputs['private_image_plan'].read_text())
        if plan['semantic_labels_or_GT_boxes_in_image_plan'] is not False:
            raise ValueError('Label-free image plan required')
        rows = sorted((r for r in plan['image_metadata'] if r['video_id'] in fit),
                      key=lambda r: (r['video_id'], r['frame_index'], r['id']))[:2]
        if len(rows) != 2:
            raise ValueError('Exactly two fit images required')
        repo = ROOT / common['local_repository']
        if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode().strip() != common['upstream_commit']
                or subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip()):
            raise ValueError('Pinned clean encoder repository required')
        source_hashes, sparse_absent, tree_python_files = verified_encoder_sources(repo, common['upstream_commit'])
        checkpoint = ROOT / common['checkpoint']
        if sha256_file(checkpoint) != common['checkpoint_sha256']:
            raise ValueError('Pinned encoder weights hash mismatch')
        guard()
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
        from scripts.trackocd_v2.build_common_features import crop_box
        sys.path.insert(0, str(repo))
        from dinov2.hub.backbones import dinov2_vitb14
        torch.set_num_threads(1); torch.manual_seed(1027)
        # No model download or dataset access is allowed after registration.
        frame_root = Path(cfg['frame_root']).resolve()
        train_root = (frame_root / 'train').resolve()
        selected_paths = {(frame_root / r['file_name']).resolve() for r in rows}
        if any(not p.is_relative_to(train_root) for p in selected_paths):
            raise ValueError('Selected paths must remain in Train root')
        def barrier(event, values):
            if event == 'socket.connect':
                raise PermissionError('No network in frozen inference')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                p = Path(os.fsdecode(values[0])).resolve()
                if (p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or p.is_relative_to(frame_root)
                        or p.is_relative_to(train_root)) and p not in selected_paths:
                    raise PermissionError('Only two selected fit pixels, no other dataset opens')
        sys.addaudithook(barrier)
        model = dinov2_vitb14(pretrained=False)
        imported_encoder_modules = 0
        for name, module in list(sys.modules.items()):
            if name != 'dinov2' and not name.startswith('dinov2.'):
                continue
            file = getattr(module, '__file__', None)
            if file is None:
                continue
            actual_path = Path(file).resolve()
            if not actual_path.is_relative_to(repo.resolve()):
                raise ValueError('Encoder module resolved outside pinned repository')
            relative = actual_path.relative_to(repo.resolve()).as_posix()
            if relative not in source_hashes or sha256_file(actual_path) != source_hashes[relative]:
                raise ValueError('Imported encoder module is not exact verified tracked source')
            imported_encoder_modules += 1
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        model.load_state_dict(state, strict=True); del state
        model.eval().requires_grad_(False)
        before = digest(model)
        transform = transforms.Compose([transforms.Resize(tuple(common['resize']), interpolation=Image.Resampling.BILINEAR),
            transforms.ToTensor(), transforms.Normalize(common['normalization_mean'], common['normalization_std'])])
        outputs, pixel_hashes = [], []
        first_tensor = None
        for row in rows:
            guard()
            path = (frame_root / row['file_name']).resolve()
            prior_stat = path.stat(); pixel_hashes.append(sha256_file(path))
            with Image.open(path) as raw:
                rgb = raw.convert('RGB'); rgb.load()
                if rgb.size != (row['width'], row['height']):
                    raise ValueError('Decoded shape differs from scalar metadata')
                # Explicit whole-image box: validates backbone/current RGB,
                # NOT GT/predicted boxes, multi-prefix crops or trajectory quality.
                tensor = transform(crop_box(rgb, [0, 0, rgb.width, rgb.height], common['crop_context_fraction_per_side']))
            with torch.inference_mode():
                feature = torch.nn.functional.normalize(model.forward_features(tensor[None])['x_norm_clstoken'], dim=-1)
            if feature.shape != (1, 768) or not torch.isfinite(feature).all() or abs(feature.norm().item() - 1) > 1e-5:
                raise ValueError('Finite unit common768 descriptor required')
            outputs.append(feature.numpy().copy())
            if first_tensor is None: first_tensor = tensor.clone()
            if (path.stat().st_size, path.stat().st_mtime_ns) != (prior_stat.st_size, prior_stat.st_mtime_ns):
                raise ValueError('Selected pixels changed during inference')
            guard()
        with torch.inference_mode():
            repeat = torch.nn.functional.normalize(model.forward_features(first_tensor[None])['x_norm_clstoken'], dim=-1)
        if not np.array_equal(repeat.numpy(), outputs[0]) or digest(model) != before or any(p.requires_grad for p in model.parameters()):
            raise ValueError('CPU repeat/frozen parameters differ')
        if torch.cuda.is_initialized():
            raise ValueError('CPU-only smoke must not initialize CUDA')
        guard()
        private = out / 'private_two_fit_descriptors.npz'
        np.savez(private, visual=np.concatenate(outputs))
        result = {'status': 'VERIFIED_TWO_FIT_IMAGE_CPU_FROZEN_BACKBONE_NOT_TRACK_FEATURES',
            'live_remote_registration_proof': proof, 'images_decoded': 2, 'actual_forward_calls': 3,
            'descriptor_shape': [2, 768], 'CPU_repeat_exact': True, 'frozen_parameters_unchanged': True,
            'frozen_parameter_sha256': before, 'checkpoint_sha256': common['checkpoint_sha256'],
            'encoder_repository_commit': common['upstream_commit'], 'verified_tracked_encoder_python_files': len(source_hashes),
            'verified_actual_imported_encoder_modules': imported_encoder_modules,
            'encoder_Git_tree_python_entries': tree_python_files, 'sparse_absent_python_paths_not_imported': sparse_absent,
            'torch_version': str(torch.__version__),
            'encoder_source_manifest_sha256': hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
            'selected_pixel_sha256': pixel_hashes, 'GT_class_role_text_or_ID_in_forward': False,
            'development_or_final_pixels_read': False, 'GT_or_predicted_crop_features_verified': False,
            'full_cache_or_training_started': False, 'new_protocol_frozen': False,
            'private_descriptor': {'bytes': private.stat().st_size, 'sha256': sha256_file(private)},
            'resources': {'wall_seconds_excluding_network': time.monotonic() - started,
                          'peak_guarded_RSS_bytes': peak, 'training_GPU_hours': 0, 'CUDA_initialized': False}}
        guard(); atomic_json(out / 'result.json', result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False})
        raise


def main():
    global SOURCES
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True); parser.add_argument('--git-proxy', default='')
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--attempt', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--config', choices=(SOURCES[0], 'configs/trackocd_cvpr/selected_frozen_encoder_smoke_v2_sparse.json',
                        'configs/trackocd_cvpr/selected_frozen_encoder_smoke_v3_terminal.json'), default=SOURCES[0])
    args = parser.parse_args(); SOURCES = (args.config, *SOURCES[1:])
    cfg = json.loads((ROOT / args.config).read_text())
    safety = json.loads((ROOT / SOURCES[-1]).read_text())
    if args.worker: worker(args, cfg, safety); return 0
    return supervise(args, cfg, safety)


if __name__ == '__main__':
    sys.exit(main())
