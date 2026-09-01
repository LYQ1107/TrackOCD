"""Build the legal FULL TAO TRAIN trajectory pool for Phase 6D.

Source: TAO TRAIN annotation file (identity + box geometry only; category
labels are used only for statistics and for the supported-known subset, never
as novel supervision).

Outputs:
  data/caches/features/dinov2/full_tao_train/{video_id}_{track_id}.json
  outputs/iclr27_phase6d/assets/full_tao_tracks.npz
  outputs/iclr27_phase6d/assets/full_tao_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(ROOT))

TAO_TRAIN_ANN = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/train.json")
FRAMES_ROOT = ROOT / "data" / "raw" / "tao" / "frames"
CACHE_DIR = ROOT / "data" / "caches" / "features" / "dinov2" / "full_tao_train"
KNOWN_IDS = set(json.loads(
    (ROOT / "data" / "trackocd_v1" / "pure" / "splits" / "supported_known_ids.json").read_text()))
OUT = ROOT / "outputs" / "iclr27_phase6d" / "assets"


def load_tracks():
    d = json.loads(TAO_TRAIN_ANN.read_text())
    img_by_id = {im["id"]: im for im in d["images"]}
    groups = defaultdict(list)
    for a in d["annotations"]:
        if a.get("iscrowd"):
            continue
        groups[(a["video_id"], a["track_id"])].append(a)
    tracks = []
    for (vid, tid), anns in groups.items():
        anns.sort(key=lambda a: img_by_id[a["image_id"]]["frame_index"])
        rows = []
        for a in anns:
            im = img_by_id[a["image_id"]]
            x, y, w, h = [float(v) for v in a["bbox"]]
            rows.append((im["file_name"], [x, y, x + w, y + h]))
        tracks.append({
            "sample_id": f"{vid}_{tid}",
            "video_id": vid,
            "track_id": tid,
            "category_id": int(anns[0]["category_id"]),
            "image_paths": [r[0] for r in rows],
            "boxes_xyxy": [r[1] for r in rows],
            "num_frames": len(rows),
        })
    return tracks


def stats(tracks):
    known = [t for t in tracks if t["category_id"] in KNOWN_IDS]
    novel = [t for t in tracks if t["category_id"] not in KNOWN_IDS]
    lens = [t["num_frames"] for t in tracks]
    by_cat = defaultdict(list)
    for t in tracks:
        by_cat[t["category_id"]].append(t["sample_id"])
    pairs = sum(len(v) * (len(v) - 1) // 2 for v in by_cat.values())
    nov_pairs = sum(len(v) * (len(v) - 1) // 2 for c, v in by_cat.items()
                    if c not in KNOWN_IDS)
    return {
        "videos": len({t["video_id"] for t in tracks}),
        "frames": sum(lens),
        "tracks": len(tracks),
        "track_len_mean": float(np.mean(lens)),
        "track_len_median": float(np.median(lens)),
        "track_len_min": int(min(lens)),
        "track_len_max": int(max(lens)),
        "known_tracks": len(known),
        "unlabeled_tracks": len(novel),
        "known_categories": len({t["category_id"] for t in known}),
        "unlabeled_categories": len({t["category_id"] for t in novel}),
        "same_category_track_pairs": pairs,
        "novel_same_category_track_pairs": nov_pairs,
        "feature_coverage": {
            "cached": sum(1 for t in tracks if (CACHE_DIR / f"{t['sample_id']}.json").exists()),
            "total": len(tracks),
        },
    }


def make_transform(size=518):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def sample_indices(n, max_frames=8):
    if n <= max_frames:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, max_frames).astype(int).tolist()))


def crop_bbox(img, box, context=0.10):
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = bw * (1 + 2 * context), bh * (1 + 2 * context)
    x1c, y1c = max(0, cx - nw / 2), max(0, cy - nh / 2)
    x2c, y2c = min(w, cx + nw / 2), min(h, cy + nh / 2)
    if x2c - x1c < 2 or y2c - y1c < 2:
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return img.crop((int(x1c), int(y1c), int(x2c), int(y2c)))


def extract_shard(tracks, shard, n_shards, device, max_frames=8):
    import torch
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)
    tf = make_transform()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mine = tracks[shard::n_shards]
    t0 = time.time()
    done = 0
    for t in mine:
        out = CACHE_DIR / f"{t['sample_id']}.json"
        launched = CACHE_DIR / f"{t['sample_id']}.launched"
        if out.exists() or launched.exists():
            continue
        launched.write_text(str(os.getpid()))
        try:
            idx = sample_indices(len(t["image_paths"]), max_frames)
            tensors = []
            valid = 0
            for i in idx:
                p = FRAMES_ROOT / t["image_paths"][i]
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue
                crop = crop_bbox(img, t["boxes_xyxy"][i])
                if min(crop.size) < 4:
                    continue
                tensors.append(tf(crop))
                valid += 1
            if not tensors:
                continue
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                f = torch.nn.functional.normalize(model(batch), dim=-1)
            f = f.cpu().numpy().astype(np.float32)
            mean = f.mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-12)
            result = {
                "sample_id": t["sample_id"],
                "video_id": t["video_id"],
                "track_id": t["track_id"],
                "category_id": t["category_id"],
                "frame_embeddings": f.astype(np.float16).tolist(),
                "mean_embedding": mean.astype(np.float16).tolist(),
                "num_valid_frames": valid,
            }
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, separators=(",", ":")))
            os.replace(tmp, out)
            done += 1
        finally:
            launched.unlink(missing_ok=True)
    print(f"shard {shard}/{n_shards}: tracks={len(mine)} done={done} "
          f"sec={time.time() - t0:.1f}", flush=True)


def assemble():
    tracks = load_tracks()
    sample_ids, labels, vids, tids, frame_feats, masks, means = (
        [], [], [], [], [], [], [])
    missing = []
    for t in tracks:
        p = CACHE_DIR / f"{t['sample_id']}.json"
        if not p.exists():
            missing.append(t["sample_id"])
            continue
        r = json.loads(p.read_text())
        ff = np.asarray(r["frame_embeddings"], dtype=np.float32)
        ff = ff / (np.linalg.norm(ff, axis=-1, keepdims=True) + 1e-12)
        buf = np.zeros((8, 768), dtype=np.float16)
        buf[: len(ff)] = ff.astype(np.float16)
        m = np.zeros((8,), dtype=np.uint8)
        m[: len(ff)] = 1
        sample_ids.append(t["sample_id"])
        labels.append(t["category_id"])
        vids.append(t["video_id"])
        tids.append(t["track_id"])
        frame_feats.append(buf)
        masks.append(m)
        means.append(np.asarray(r["mean_embedding"], dtype=np.float32))
    arr = {
        "sample_ids": np.asarray(sample_ids),
        "labels": np.asarray(labels, dtype=np.int32),
        "video_ids": np.asarray(vids, dtype=np.int32),
        "track_ids": np.asarray(tids, dtype=np.int32),
        "frame_feats": np.stack(frame_feats),
        "frame_mask": np.stack(masks),
        "mean_feats": np.stack(means),
        "is_known": np.asarray([int(c in KNOWN_IDS) for c in labels], dtype=np.uint8),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "full_tao_tracks.npz", **arr)
    st = stats(tracks)
    (OUT / "full_tao_stats.json").write_text(json.dumps(st, indent=2))
    print("assembled", len(sample_ids), "tracks, missing", len(missing))
    print(json.dumps(st, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stats", "extract", "assemble"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=8)
    args = ap.parse_args()
    tracks = load_tracks()
    if args.mode == "stats":
        print(json.dumps(stats(tracks), indent=2))
    elif args.mode == "extract":
        extract_shard(tracks, args.shard, args.n_shards, args.device,
                      args.max_frames)
    else:
        assemble()


if __name__ == "__main__":
    main()
