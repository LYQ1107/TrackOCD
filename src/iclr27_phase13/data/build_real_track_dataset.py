"""Build a compact real-video TAO track-level pretraining asset.

Phase 6D already extracted DINOv2 features from real TAO TRAIN bbox crops.
This builder joins those cached temporal crop features to the public TAO
track metadata and derives a causal, scale-normalized box-motion stream.  It
does not use MOTSynth, Q1 labels, or private Q1 annotations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
PUBLIC = ROOT / "data/tao_ow_ocd_v1/public/train_known_tracks.jsonl"
FEATURE_ROOT = ROOT / "data/caches/features/dinov2/full_tao_train"
SOURCE_NPZ = ROOT / "outputs/iclr27_phase6c/assets/known_tracks.npz"


def sample_indices(n: int, k: int) -> np.ndarray:
    if n <= k:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, k).astype(np.int64))


def motion_from_boxes(boxes: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return [normalised dx, dy, dlog-width, dlog-height] per sampled frame.

    The first row is zero because no previous frame is available.  Position
    deltas are divided by the geometric-mean size of the previous box, making
    the feature usable across TAO video resolutions and Q1 proposal streams.
    """
    b = np.asarray(boxes, dtype=np.float32)[indices]
    x1, y1, x2, y2 = b.T
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w = np.maximum(x2 - x1, 1.0)
    h = np.maximum(y2 - y1, 1.0)
    out = np.zeros((len(b), 4), dtype=np.float32)
    if len(b) > 1:
        scale = np.sqrt(w[:-1] * h[:-1])
        out[1:, 0] = (cx[1:] - cx[:-1]) / scale
        out[1:, 1] = (cy[1:] - cy[:-1]) / scale
        out[1:, 2] = np.log(w[1:] / w[:-1])
        out[1:, 3] = np.log(h[1:] / h[:-1])
    return np.clip(out, -8.0, 8.0)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/iclr27_phase13/dataset")
    ap.add_argument("--max-frames", type=int, default=8)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    source = np.load(SOURCE_NPZ)
    source_ids = [str(x) for x in source["sample_ids"]]
    source_index = {sid: i for i, sid in enumerate(source_ids)}
    records = {}
    with PUBLIC.open() as f:
        for line in f:
            r = json.loads(line)
            records[str(r["sample_id"])] = r

    features, motions, masks = [], [], []
    labels, video_ids, track_ids, sample_ids = [], [], [], []
    image_path_counts = []
    missing = []
    for sid in source_ids:
        r = records.get(sid)
        cache_path = FEATURE_ROOT / f"{sid}.json"
        if r is None or not cache_path.exists():
            missing.append(sid)
            continue
        c = json.loads(cache_path.read_text())
        ff = np.asarray(c["frame_embeddings"], dtype=np.float32)
        if ff.ndim != 2 or ff.shape[0] == 0:
            missing.append(sid)
            continue
        ff = ff[: args.max_frames]
        idx = sample_indices(len(r["image_paths"]), len(ff))
        boxes = np.asarray(r["boxes_xyxy"], dtype=np.float32)
        if len(boxes) < len(idx):
            missing.append(sid)
            continue
        mm = motion_from_boxes(boxes, idx)
        x = np.zeros((args.max_frames, ff.shape[1]), dtype=np.float32)
        m = np.zeros((args.max_frames,), dtype=np.uint8)
        x[: len(ff)] = ff
        m[: len(ff)] = 1
        mot = np.zeros((args.max_frames, 4), dtype=np.float32)
        mot[: len(mm)] = mm
        features.append(x)
        motions.append(mot)
        masks.append(m)
        labels.append(int(r["category_id"]))
        video_ids.append(int(r["video_id"]))
        track_ids.append(int(r["track_id"]))
        sample_ids.append(sid)
        image_path_counts.append(int(len(r["image_paths"])))

    if not features:
        raise RuntimeError("no real TAO tracks could be assembled")
    arrays = {
        "appearance": np.stack(features).astype(np.float32),
        "motion": np.stack(motions).astype(np.float32),
        "mask": np.stack(masks).astype(np.uint8),
        "labels": np.asarray(labels, dtype=np.int32),
        "video_ids": np.asarray(video_ids, dtype=np.int32),
        "track_ids": np.asarray(track_ids, dtype=np.int32),
        "sample_ids": np.asarray(sample_ids),
    }
    atomic_npz(out / "real_tao_tracks.npz", **arrays)
    metadata = {
        "source_dataset": "TAO TRAIN (real YFCC100M/BDD/YouTube-VOS videos)",
        "source_public_tracks": str(PUBLIC),
        "source_feature_cache": str(FEATURE_ROOT),
        "source_frame_root": str(ROOT / "data/raw/tao/frames"),
        "source_phase6d_asset": str(SOURCE_NPZ),
        "not_motsynth": True,
        "real_video_tracks": True,
        "tracks": int(len(features)),
        "videos": int(len(set(video_ids))),
        "categories": int(len(set(labels))),
        "appearance_dim": int(arrays["appearance"].shape[-1]),
        "motion_dim": 4,
        "max_frames": int(args.max_frames),
        "mean_source_frames_per_track": float(np.mean(image_path_counts)),
        "category_labels_used_for_dataset_index_only": True,
        "q1_labels_used": False,
        "private_gt_used": False,
        "physical_id_used_as_feature": False,
        "missing_records": missing,
        "motion_definition": "causal normalized box center/scale deltas",
    }
    tmp = out / "metadata.json.tmp"
    tmp.write_text(json.dumps(metadata, indent=2))
    os.replace(tmp, out / "metadata.json")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
