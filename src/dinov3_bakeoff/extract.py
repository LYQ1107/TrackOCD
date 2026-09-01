#!/usr/bin/env python3
"""Extract DINOv3 ViT-B/16 track features with the official preprocessing
(256x256 RGB, ImageNet mean/std, no augmentation), same bbox crop + frame
sampling as the DINOv2 cache."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.dinov3_bakeoff.adapter import DinoV3Adapter, write_track_cache
from src.data.tao_io import FRAMES_ROOT
from src.features.extract import crop_bbox, sample_indices

CACHE_ROOT = PROJECT_ROOT / "data" / "caches" / "features" / "dinov3_vitb16_lvd1689m"


def make_transform():
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_rows(split, stream="val_gt_track_stream.jsonl"):
    if split == "train_known":
        p = PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / "train_known_tracks.jsonl"
    else:
        p = PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / stream
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract(adapter, transform, rows, cache_dir, mode):
    done = 0
    t0 = time.time()
    for row in rows:
        out_path = cache_dir / f"{row['sample_id']}.json"
        launched = cache_dir / f"{row['sample_id']}.launched"
        if out_path.exists() or launched.exists():
            continue
        launched.write_text(str(__import__("os").getpid()))
        try:
            n = len(row["image_paths"])
            if mode == "single":
                idx = [n // 2]
            else:
                idx = sample_indices(n, 8)
            tensors = []
            valid_idx = []
            for i in idx:
                try:
                    img = Image.open(FRAMES_ROOT / row["image_paths"][i]).convert("RGB")
                except Exception:
                    continue
                crop = crop_bbox(img, row["boxes_xyxy"][i])
                if min(crop.size) < 4:
                    continue
                tensors.append(transform(crop))
                fid = row["frame_ids"][i] if "frame_ids" in row else row["image_paths"][i]
                valid_idx.append(fid)
            if not tensors:
                launched.unlink(missing_ok=True)
                continue
            embeds = adapter.embed_crops(tensors)
            write_track_cache(adapter, row, valid_idx, embeds, cache_dir, mode)
            done += 1
        finally:
            launched.unlink(missing_ok=True)
    print(f"mode={mode} tracks={len(rows)} done={done} sec={time.time()-t0:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train_known", "gt_val"], required=True)
    ap.add_argument("--mode", choices=["single", "mean"], required=True)
    ap.add_argument("--stream", default="val_gt_track_stream.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feature-mode", choices=["cls", "pooled"], default="cls")
    args = ap.parse_args()
    adapter = DinoV3Adapter(device=args.device, feature_mode=args.feature_mode)
    transform = make_transform()
    rows = load_rows(args.split, args.stream)
    if args.split == "train_known":
        cache_dir = CACHE_ROOT / "train_known"
    else:
        cache_dir = CACHE_ROOT / "gt_tracks" / args.mode
    cache_dir.mkdir(parents=True, exist_ok=True)
    extract(adapter, transform, rows, cache_dir, args.mode)


if __name__ == "__main__":
    main()
