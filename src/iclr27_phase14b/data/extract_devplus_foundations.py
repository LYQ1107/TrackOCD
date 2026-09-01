"""Extract frozen foundation features for the quarantined-Q1-free DEV+ GT view."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT')
MANIFEST = ROOT / 'outputs/iclr27_phase14b/manifests/devplus_tracks.jsonl'
FRAME_ROOT = ROOT / 'data/iclr27_phase14b/sources/tao_train_frames'


def crop_bbox(image: Image.Image, box, context=0.10):
    width, height = image.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    nw, nh = bw * (1.0 + 2.0 * context), bh * (1.0 + 2.0 * context)
    left = max(0.0, cx - nw * 0.5); top = max(0.0, cy - nh * 0.5)
    right = min(float(width), cx + nw * 0.5); bottom = min(float(height), cy + nh * 0.5)
    if right - left < 2.0 or bottom - top < 2.0:
        left, top, right, bottom = max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)
    return image.crop((int(left), int(top), int(right), int(bottom)))


def sample_indices(length, max_frames=8):
    # The benchmark is prefix-causal: the canonical cache contains the first
    # max_frames observations, never an evenly spaced sample from the full
    # track (which would leak future frames into an early prefix).
    return list(range(min(length, max_frames)))


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def load_rows():
    rows = []
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('split') == 'devplus' and row.get('role') == 'devplus_novel':
                rows.append(row)
    rows.sort(key=lambda r: int(r['chronological_position']))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', choices=['dinov2', 'clip', 'dinov3'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()
    rows = load_rows()
    if not rows:
        raise RuntimeError('no DEV+ rows')
    device = torch.device(args.device)
    if args.encoder == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        model.eval().to(device)
        dim = int(model.num_features)
        transform = transforms.Compose([
            transforms.Resize((518, 518), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        def encode(batch):
            with torch.no_grad():
                return torch.nn.functional.normalize(model(batch), dim=-1).cpu().numpy().astype(np.float32)
        model_id = 'DINOv2 ViT-B/14 image encoder'
        flags = {'q1_labels_used': False, 'private_gt_used': False, 'future_used': False, 'physical_id_used': False}
    elif args.encoder == 'clip':
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained=str(ROOT / 'data/iclr27_phase14b/checkpoints/clip_vitb32_openai.pt'),
            device=device, weights_only=False)
        model.eval()
        dim = int(model.visual.output_dim)
        transform = preprocess
        def encode(batch):
            with torch.no_grad():
                return torch.nn.functional.normalize(model.encode_image(batch), dim=-1).cpu().numpy().astype(np.float32)
        model_id = 'OpenAI CLIP ViT-B/32 visual encoder'
        flags = {'q1_labels_used': False, 'private_gt_used': False, 'future_used': False, 'physical_id_used': False}
    else:
        from src.dinov3_bakeoff.adapter import DinoV3Adapter
        adapter = DinoV3Adapter(device=args.device, feature_mode='cls')
        dim = int(adapter.feature_dim)
        transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        def encode(batch):
            with torch.no_grad():
                return adapter.embed_crops([x for x in batch])
        model_id = adapter.model_id + ' CLS'
        flags = {'q1_labels_used': False, 'private_gt_used': False, 'future_used': False, 'physical_id_used': False}

    max_frames = 16
    track_features = np.zeros((len(rows), max_frames, dim), dtype=np.float32)
    track_mask = np.zeros((len(rows), max_frames), dtype=np.uint8)
    failures = []
    pending = []
    pending_keys = []
    pending_slots = []

    def flush():
        nonlocal pending, pending_keys, pending_slots
        if not pending:
            return
        batch = torch.stack(pending).to(device)
        values = encode(batch)
        for index, key, slot in zip(pending_slots, pending_keys, values):
            track_features[index, key] = slot
            track_mask[index, key] = 1
        pending, pending_keys, pending_slots = [], [], []

    for index, row in enumerate(rows):
        indices = sample_indices(len(row['image_paths']), max_frames)
        for slot, frame_index in enumerate(indices):
            path = FRAME_ROOT / row['image_paths'][frame_index]
            try:
                with Image.open(path) as image:
                    tensor = transform(crop_bbox(image.convert('RGB'), row['boxes_xyxy'][frame_index]))
                pending.append(tensor); pending_keys.append(slot); pending_slots.append(index)
                if len(pending) >= args.batch_size:
                    flush()
            except Exception as exc:
                failures.append({'sample_id': row['sample_id'], 'frame_index': int(frame_index), 'reason': type(exc).__name__})
        if index % 20 == 0:
            print(f'prepared {index + 1}/{len(rows)}', flush=True)
    flush()
    mean = track_features.sum(axis=1) / np.maximum(track_mask.sum(axis=1, keepdims=True), 1)
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    atomic_npz(Path(args.out), sample_keys=np.asarray([r['sample_id'] for r in rows]),
               frame_features=track_features, frame_mask=track_mask, mean_features=mean)
    meta = {
        'representation': model_id, 'encoder': args.encoder, 'rows': len(rows),
        'shape': list(track_features.shape), 'failed_frames': failures,
        'prefix_sampling': 'first_min(16, track_length) frames; no evenly-spaced future samples',
        'manifest': str(MANIFEST), 'q1_used': False, 'feature_selection_used': False,
        'model_selection_used': False, **flags,
    }
    Path(args.out).with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
