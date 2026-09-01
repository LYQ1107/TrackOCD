"""Build trajectory-level DINOv2 training datasets for Phase 6C.

Known pool: 2196 TAO TRAIN supported-known GT tracks (per-frame DINOv2
features already cached under data/caches/features/dinov2/train_known_mean).
Unlabeled pool: Phase 4T TRAIN stream (model-generated trajectories with
DINOv2 per-frame features in outputs/iclr27_phase4t/train_stream/feats.npz).
GT columns in the TRAIN stream are ignored; no novel labels are used.

Outputs:
  outputs/iclr27_phase6c/assets/known_tracks.npz
  outputs/iclr27_phase6c/assets/unlabeled_tracks.npz
  outputs/iclr27_phase6c/assets/pca.npz
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
KNOWN_JSONL = ROOT / "data" / "tao_ow_ocd_v1" / "public" / "train_known_tracks.jsonl"
KNOWN_FEAT_DIR = ROOT / "data" / "caches" / "features" / "dinov2" / "train_known_mean"
UNLABELED_CSV = ROOT / "outputs" / "iclr27_phase4t" / "train_stream" / "proposals.csv"
UNLABELED_FEATS = ROOT / "outputs" / "iclr27_phase4t" / "train_stream" / "feats.npz"
OUT = ROOT / "outputs" / "iclr27_phase6c" / "assets"


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def sample_indices(n: int, max_frames: int = 8):
    if n <= max_frames:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, max_frames).astype(int).tolist()))


def build_known():
    rows = []
    with open(KNOWN_JSONL) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    sample_ids, labels, frame_feats, frame_mask, mean_feats, n_frames = (
        [], [], [], [], [], []
    )
    missing = 0
    for r in rows:
        p = KNOWN_FEAT_DIR / f"{r['sample_id']}.json"
        if not p.exists():
            missing += 1
            continue
        c = json.loads(p.read_text())
        ff = np.asarray(c["frame_embeddings"], dtype=np.float32)
        ff = l2norm(ff)
        T = ff.shape[0]
        idx = sample_indices(T, 8)
        ff = ff[idx]
        m = np.zeros((8,), dtype=np.uint8)
        m[: len(idx)] = 1
        buf = np.zeros((8, 768), dtype=np.float16)
        buf[: len(ff)] = ff.astype(np.float16)
        sample_ids.append(r["sample_id"])
        labels.append(int(r["category_id"]))
        frame_feats.append(buf)
        frame_mask.append(m)
        mean_feats.append(np.asarray(c["mean_embedding"], dtype=np.float32))
        n_frames.append(len(idx))
    arr = {
        "sample_ids": np.asarray(sample_ids),
        "labels": np.asarray(labels, dtype=np.int32),
        "frame_feats": np.stack(frame_feats),
        "frame_mask": np.stack(frame_mask),
        "mean_feats": l2norm(np.stack(mean_feats)).astype(np.float32),
        "n_frames": np.asarray(n_frames, dtype=np.int32),
    }
    np.savez_compressed(OUT / "known_tracks.npz", **arr)
    print(f"known: {len(sample_ids)} tracks, {len(set(labels))} classes, "
          f"missing_cache={missing}")


def build_unlabeled(min_len=3, max_frames=8):
    df = pd.read_csv(UNLABELED_CSV)
    feats = np.load(UNLABELED_FEATS)["feats"].astype(np.float32)
    assert len(df) == len(feats)
    feats = l2norm(feats)
    groups = defaultdict(list)
    for i, r in df.iterrows():
        groups[(int(r["video_id"]), int(r["track_id"]))].append(i)
    keys = [k for k, idxs in groups.items() if len(idxs) >= min_len]
    track_vids, track_tids = [], []
    frame_feats, frame_mask, mean_feats, n_frames = [], [], [], []
    for k in keys:
        idxs = groups[k]
        si = sample_indices(len(idxs), max_frames)
        sel = [idxs[j] for j in si]
        ff = feats[sel]
        m = np.zeros((max_frames,), dtype=np.uint8)
        m[: len(sel)] = 1
        buf = np.zeros((max_frames, 768), dtype=np.float16)
        buf[: len(ff)] = ff.astype(np.float16)
        track_vids.append(k[0])
        track_tids.append(k[1])
        frame_feats.append(buf)
        frame_mask.append(m)
        mean_feats.append(ff.mean(axis=0))
        n_frames.append(len(sel))
    arr = {
        "video_ids": np.asarray(track_vids, dtype=np.int32),
        "track_ids": np.asarray(track_tids, dtype=np.int32),
        "frame_feats": np.stack(frame_feats),
        "frame_mask": np.stack(frame_mask),
        "mean_feats": l2norm(np.stack(mean_feats)).astype(np.float32),
        "n_frames": np.asarray(n_frames, dtype=np.int32),
    }
    np.savez_compressed(OUT / "unlabeled_tracks.npz", **arr)
    print(f"unlabeled: {len(keys)} tracks (>= {min_len} rows), "
          f"{len(keys) * max_frames} frame slots")


def build_unlabeled_gt(min_len=3, max_frames=8):
    """Clean unlabeled pool: TRAIN stream rows grouped by GT physical track
    id (identity only; gt_role/gt_category_id are NOT used for training)."""
    df = pd.read_csv(UNLABELED_CSV)
    feats = np.load(UNLABELED_FEATS)["feats"].astype(np.float32)
    feats = l2norm(feats)
    mask = (df["gt_track_id"].values >= 0)
    pos = np.flatnonzero(mask)
    df = df[mask].reset_index(drop=True)
    feats = feats[pos]
    groups = defaultdict(list)
    for i, r in df.iterrows():
        groups[(int(r["video_id"]), int(r["gt_track_id"]))].append(i)
    keys = [k for k, idxs in groups.items() if len(idxs) >= min_len]
    track_vids, track_tids = [], []
    frame_feats, frame_mask, mean_feats, n_frames = [], [], [], []
    for k in keys:
        idxs = groups[k]
        si = sample_indices(len(idxs), max_frames)
        sel = [idxs[j] for j in si]
        ff = feats[sel]
        m = np.zeros((max_frames,), dtype=np.uint8)
        m[: len(sel)] = 1
        buf = np.zeros((max_frames, 768), dtype=np.float16)
        buf[: len(ff)] = ff.astype(np.float16)
        track_vids.append(k[0])
        track_tids.append(k[1])
        frame_feats.append(buf)
        frame_mask.append(m)
        mean_feats.append(ff.mean(axis=0))
        n_frames.append(len(sel))
    arr = {
        "video_ids": np.asarray(track_vids, dtype=np.int32),
        "track_ids": np.asarray(track_tids, dtype=np.int32),
        "frame_feats": np.stack(frame_feats),
        "frame_mask": np.stack(frame_mask),
        "mean_feats": l2norm(np.stack(mean_feats)).astype(np.float32),
        "n_frames": np.asarray(n_frames, dtype=np.int32),
    }
    np.savez_compressed(OUT / "unlabeled_gt_tracks.npz", **arr)
    print(f"unlabeled_gt: {len(keys)} tracks (>= {min_len} rows)")


def build_pca(max_tracks=30000):
    k = np.load(OUT / "known_tracks.npz")
    u = np.load(OUT / "unlabeled_tracks.npz")
    X = [k["mean_feats"]]
    if len(u["mean_feats"]) <= max_tracks:
        X.append(u["mean_feats"])
    else:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(u["mean_feats"]), max_tracks, replace=False)
        X.append(u["mean_feats"][idx])
    X = np.concatenate(X, axis=0).astype(np.float32)
    pca = PCA(n_components=128, random_state=0, svd_solver="randomized")
    pca.fit(X)
    np.savez_compressed(
        OUT / "pca.npz",
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        n_samples=len(X),
    )
    print(f"pca fit on {len(X)} track means, explained "
          f"{pca.explained_variance_ratio_.sum():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-unlabeled-len", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    build_known()
    build_unlabeled(min_len=args.min_unlabeled_len, max_frames=args.max_frames)
    build_unlabeled_gt(min_len=args.min_unlabeled_len, max_frames=args.max_frames)
    build_pca()


if __name__ == "__main__":
    main()
