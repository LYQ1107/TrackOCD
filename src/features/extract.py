#!/usr/bin/env python3
"""Frozen feature extraction for TAO-OW TrackOCD v1.

Extracts bbox-cropped frame embeddings (DINOv2 / CLIP) for GT tracks and
(later) predicted tracks. Every track is written atomically and can be resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tao_io import FRAMES_ROOT, atomic_write_text


def load_dinov2():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().cuda()
    return model, 518


def load_clip():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device="cuda"
    )
    model.eval().cuda()
    return model, 224, preprocess


def make_crop_transform(size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def sample_indices(n: int, max_frames: int = 8):
    if n <= max_frames:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, max_frames).astype(int).tolist()))


def sample_indices_scored(n: int, max_frames: int, scores, areas):
    """Time-uniform sampling that prefers high detection scores and avoids
    extreme-area frames (used for predicted tracks)."""
    if n <= max_frames:
        return list(range(n))
    valid = []
    for i in range(n):
        area = areas[i]
        if area is None or area <= 0:
            continue
        valid.append(i)
    if not valid:
        valid = list(range(n))
    # drop extreme areas (top/bottom 5% within the track)
    area_arr = np.asarray([areas[i] for i in valid], dtype=np.float64)
    lo, hi = np.percentile(area_arr, 5), np.percentile(area_arr, 95)
    valid = [i for i in valid if lo <= areas[i] <= hi] or valid
    # one frame per time bin, highest score within bin
    chosen = []
    for b in range(max_frames):
        start = int(np.floor(b * n / max_frames))
        end = int(np.ceil((b + 1) * n / max_frames))
        cands = [i for i in valid if start <= i < end]
        if cands:
            chosen.append(max(cands, key=lambda i: scores[i]))
    if len(chosen) < max_frames:
        rest = [i for i in valid if i not in chosen]
        rest.sort(key=lambda i: scores[i], reverse=True)
        chosen.extend(rest[: max_frames - len(chosen)])
    return sorted(chosen)


def crop_bbox(img: Image.Image, box_xyxy, context=0.10):
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = bw * (1 + 2 * context), bh * (1 + 2 * context)
    x1c = max(0, cx - nw / 2)
    y1c = max(0, cy - nh / 2)
    x2c = min(w, cx + nw / 2)
    y2c = min(h, cy + nh / 2)
    if x2c - x1c < 2 or y2c - y1c < 2:
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return img.crop((int(x1c), int(y1c), int(x2c), int(y2c)))


def encode_images(model, tensors, clip_preprocess=None):
    with torch.no_grad():
        if clip_preprocess is None:
            feats = model(tensors)
        else:
            # open_clip model forward expects normalized tensor directly
            feats = model.encode_image(tensors)
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.cpu().numpy().astype(np.float32)


def extract_track_features(
    row,
    model,
    size,
    transform,
    cache_dir,
    clip_preprocess=None,
    mode="mean",
    sampling="uniform",
):
    """Extract features for one track. mode in {'mean','single'}."""
    sample_id = row["sample_id"]
    out_path = cache_dir / f"{sample_id}.json"
    launched = cache_dir / f"{sample_id}.launched"
    if out_path.exists():
        return False
    if launched.exists():
        return False
    launched.write_text(str(os.getpid()))

    image_paths = row["image_paths"]
    boxes = row["boxes_xyxy"]
    if mode == "single":
        idx = [len(image_paths) // 2]
    elif sampling == "score":
        idx = sample_indices_scored(
            len(image_paths), 8, row.get("scores", [1.0] * len(image_paths)), row.get("areas", [1] * len(image_paths))
        )
    else:
        idx = sample_indices(len(image_paths), 8)

    tensors = []
    valid_idx = []
    valid = 0
    for i in idx:
        p = FRAMES_ROOT / image_paths[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        crop = crop_bbox(img, boxes[i])
        if min(crop.size) < 4:
            continue
        t = transform(crop).unsqueeze(0).cuda()
        tensors.append(t)
        valid_idx.append(i)
        valid += 1

    if not tensors:
        launched.unlink(missing_ok=True)
        raise RuntimeError(f"no valid frames for {sample_id}")

    batch = torch.cat(tensors, dim=0)
    with torch.no_grad():
        if clip_preprocess is None:
            f = model(batch)
        else:
            f = model.encode_image(batch)
        f = torch.nn.functional.normalize(f, dim=-1)
    embeds = f.cpu().numpy().astype(np.float32)

    embeds = np.asarray(embeds)
    mean = embeds.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-12)
    result = {
        "sample_id": sample_id,
        "frame_embeddings": embeds.astype(np.float16).tolist(),
        "mean_embedding": mean.astype(np.float16).tolist(),
        "num_valid_frames": valid,
    }
    atomic_write_text(out_path, json.dumps(result, separators=(",", ":")))
    launched.unlink(missing_ok=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["dinov2", "clip"], required=True)
    ap.add_argument("--split", choices=["train_known", "gt_val", "pred_val"], required=True)
    ap.add_argument("--mode", choices=["mean", "single"], default="mean")
    ap.add_argument("--sampling", choices=["uniform", "score"], default="uniform")
    ap.add_argument("--stream", default="val_gt_track_stream.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if args.encoder == "dinov2":
        model, size = load_dinov2()
        clip_preprocess = None
    else:
        model, size, clip_preprocess = load_clip()
    transform = make_crop_transform(size)

    if args.split == "train_known":
        stream_path = PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / "train_known_tracks.jsonl"
        subdir = "train_known"
    elif args.split == "gt_val":
        stream_path = PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / args.stream
        subdir = "gt_tracks"
    else:
        stream_path = PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / args.stream
        subdir = "pred_tracks"

    cache_dir = PROJECT_ROOT / "data" / "caches" / "features" / args.encoder / f"{subdir}_{args.mode}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(stream_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    t0 = time.time()
    done = 0
    for row in rows:
        try:
            extract_track_features(
                row, model, size, transform, cache_dir, clip_preprocess, args.mode, args.sampling
            )
            done += 1
        except Exception as e:
            print(f"ERROR {row['sample_id']}: {e}", flush=True)
    print(
        f"encoder={args.encoder} split={args.split} mode={args.mode} "
        f"tracks={len(rows)} done={done} sec={time.time()-t0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
