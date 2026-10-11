"""Current-image-only shared DINO protocol; no labels/routing to forward."""
from collections import defaultdict
import os
from pathlib import Path
import numpy as np
from src.trackocd_v2.io import atomic_json, sha256_file


class FrozenTrainEncoder:
    def __init__(self, root, cfg, common):
        import sys
        import torch
        from torchvision import transforms
        from PIL import Image
        from scripts.trackocd_core.run_masa_physical_qualification import frozen_state_digest
        sys.path.insert(0, str(root / common['local_repository']))
        from dinov2.hub.backbones import dinov2_vitb14
        self.root, self.cfg, self.common, self.torch = root, cfg, common, torch
        self.digest = frozen_state_digest
        torch.set_num_threads(1)
        torch.manual_seed(1027)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model = dinov2_vitb14(pretrained=False)
        state = torch.load(root / common['checkpoint'], map_location='cpu', mmap=True, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.eval().requires_grad_(False)
        self.before = self.digest(self.model)
        torch.cuda.set_per_process_memory_fraction(min(1., cfg['GPU_peak_bytes'] / torch.cuda.get_device_properties(0).total_memory))
        self.model.to('cuda:0')
        torch.cuda.reset_peak_memory_stats()
        self.transform = transforms.Compose([
            transforms.Resize(tuple(common['resize']), interpolation=Image.Resampling.BILINEAR),
            transforms.ToTensor(), transforms.Normalize(common['normalization_mean'], common['normalization_std'])])

    def video(self, video, physical_path, guard, progress_path):
        from PIL import Image
        from scripts.trackocd_v2.build_common_features import crop_box
        from scripts.trackocd_cvpr.prepare_train_input_counterfactual import train_image
        from src.trackocd_core.limited_masa_features import select_all_prefixes
        from src.trackocd_cvpr.prefix_interface_checks import check_prefix_tails
        cfg, torch = self.cfg, self.torch
        forwards, deduplicated, encoded_images = 0, 0, 0
        with np.load(physical_path, allow_pickle=False) as a:
            rows, count = select_all_prefixes(video, a, cfg['maximum_observations'])
            visual = np.full((count, 768), np.nan, np.float16)
            geometry, quality = np.empty((count, 4), np.float32), np.empty(count, np.float32)
            frames, images = np.empty(count, np.int64), np.empty(count, np.int64)
            grouped = defaultdict(list)
            for row in rows:
                for j, observation in enumerate(row['observations']):
                    grouped[observation['image_position']].append((row['observation_offset'] + j, observation['source_offset']))
            for position, locations in sorted(grouped.items()):
                guard()
                image = video['images'][position]
                path = train_image(Path(cfg['frame_root']), image['image_path'])
                if path.stat().st_size != image['image_bytes'] or sha256_file(path) != image['image_sha256']:
                    raise ValueError('Registered current Train pixels changed')
                stat_before = path.stat()
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
                        guard()
                        if torch.cuda.max_memory_reserved() > cfg['GPU_peak_bytes'] or torch.cuda.mem_get_info()[0] < cfg['GPU_free_reserve_bytes']:
                            raise RuntimeError('Frozen DINO GPU ceiling/reserve')
                        chosen = boxes[first:first + cfg['batch_size']]
                        tensors = [self.transform(crop_box(rgb, box, self.common['crop_context_fraction_per_side'])) for box in chosen]
                        with torch.inference_mode():
                            z = self.model.forward_features(torch.stack(tensors).to('cuda:0'))['x_norm_clstoken']
                            z = torch.nn.functional.normalize(z, dim=-1).cpu().numpy().astype(np.float16)
                        for box, value in zip(chosen, z):
                            for offset in unique[box]:
                                visual[offset] = value
                        forwards += 1
                if (path.stat().st_size, path.stat().st_mtime_ns) != (stat_before.st_size, stat_before.st_mtime_ns):
                    raise ValueError('Current image changed during encoding')
                encoded_images += 1
                atomic_json(progress_path, {'status': 'RUNNING_BOUNDED_TRAIN_DINO_FEATURES',
                    'video_id': video['video_id'], 'image_position': position, 'video_batch_forwards': forwards, 'owned_pid': os.getpid()})
        if not np.isfinite(visual).all() or self.digest(self.model) != self.before or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Incomplete/changed frozen encoder')
        offsets = np.asarray([r['observation_offset'] for r in rows] + [count], np.int64)
        checks = check_prefix_tails(visual, geometry, quality, frames, offsets)
        guard()
        return {'visual': visual, 'geometry': geometry, 'quality': quality, 'frame_index': frames, 'image_id': images,
                'track_id': np.asarray([r['physical_track_id'] for r in rows], np.int64), 'offsets': offsets}, {
            'tracks': len(rows), 'observations': count, 'singletons': sum(len(r['observations']) == 1 for r in rows),
            'batch_forwards': forwards, 'deduplicated_identical_current_image_boxes': deduplicated, 'images_encoded': encoded_images,
            'prefix_tail_mutation': checks, 'frozen_parameter_sha256_before_and_after': self.before,
            'frozen_parameters_unchanged': True, 'GT_or_role_or_text_in_forward': False}
