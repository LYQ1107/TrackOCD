"""Build legal trajectory-level proxy-novel episodes for Phase 7A.

Source: Phase 4T TRAIN stream (model-generated physical trajectories with
DINOv2 per-frame features). Only two row groups are used:
  - gt_role == "known" : supported-known GT supervision (class labels legal);
  - gt_role == "fp"    : unlabeled low-quality rows (memory dynamics only,
                         no supervised loss).
Rows with gt_role == "novel_role" are excluded entirely so that no true
novel identity can leak into training.

The 47 supported-known classes present in the stream are split by class:
  - known         : anchors visible in train and val (target KNOWN);
  - novel_train   : anchors hidden, pseudo-novel in train;
  - novel_val     : anchors visible in train, hidden only in val episodes
                    (held-out pseudo-novel classes for checkpoint selection).

All rows are kept in chronological order inside each video; memory is reset
at chunk boundaries (CHUNK_VIDEOS videos) during training and val replay.
No true novel GT, no Q1/Q2 labels, no future information is used.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
CSV = ROOT / "outputs/iclr27_phase7a/assets/p4t_dinov2_subset.csv"
FEATS = ROOT / "outputs/iclr27_phase7a/assets/p4t_dinov2/feats.npz"
OUT = ROOT / "outputs/iclr27_phase7a/assets"


def parse_bbox(s):
    v = json.loads(s)
    return [float(x) for x in v]


def build(seed: int = 1027, val_frac: float = 0.2):
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    feats = np.load(FEATS)["feats"].astype(np.float32)
    assert len(df) == len(feats)

    keep = df["gt_role"].isin(("known", "fp"))
    df = df[keep].reset_index(drop=True)
    feats = feats[keep.to_numpy()]

    # L2 normalize per-row features (same recipe as Phase 6C).
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

    known = df[df["gt_role"] == "known"]
    classes = sorted(int(c) for c in known["gt_category_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(classes)
    n_half = len(classes) // 3
    known_cls = set(classes[: len(classes) - 2 * n_half])
    novel_train = set(classes[len(classes) - 2 * n_half: len(classes) - n_half])
    novel_val = set(classes[len(classes) - n_half:])
    split = {
        "known": sorted(known_cls),
        "novel_train": sorted(novel_train),
        "novel_val": sorted(novel_val),
        "all": classes,
    }

    videos = sorted(int(v) for v in df["video_id"].unique())
    rng2 = random.Random(seed + 1)
    rng2.shuffle(videos)
    n_val = max(1, int(round(len(videos) * val_frac)))
    val_videos = set(videos[:n_val])
    train_videos = [v for v in videos if v not in val_videos]

    rows = []
    for v in train_videos:
        m = df["video_id"] == v
        sub = df[m].copy()
        sub["_orig"] = np.flatnonzero(m.to_numpy())
        sub = sub.sort_values(
            ["frame_id", "image_id", "track_id"], kind="stable")
        rows.append(sub)
    train = pd.concat(rows, ignore_index=True)
    tr_feats = feats[train["_orig"].to_numpy(dtype=np.int64)]

    rows = []
    for v in sorted(val_videos):
        m = df["video_id"] == v
        sub = df[m].copy()
        sub["_orig"] = np.flatnonzero(m.to_numpy())
        sub = sub.sort_values(
            ["frame_id", "image_id", "track_id"], kind="stable")
        rows.append(sub)
    val = pd.concat(rows, ignore_index=True)
    va_feats = feats[val["_orig"].to_numpy(dtype=np.int64)]

    def pack(sub, fts):
        bbox = np.asarray([parse_bbox(s) for s in sub["bbox_xyxy"]],
                          dtype=np.float32)
        labels = np.where(sub["gt_role"].to_numpy() == "known",
                          sub["gt_category_id"].astype(np.int32).to_numpy(),
                          -1)
        return {
            "video_ids": sub["video_id"].to_numpy(dtype=np.int32),
            "frame_ids": sub["frame_id"].to_numpy(dtype=np.int32),
            "proposal_local_ids": sub["proposal_local_id"].to_numpy(
                dtype=np.int32) if "proposal_local_id" in sub else np.zeros(
                    len(sub), dtype=np.int32),
            "track_ids": sub["track_id"].to_numpy(dtype=np.int32),
            "gt_role": (sub["gt_role"].to_numpy() == "known").astype(np.uint8),
            "gt_category_id": labels,
            "score": sub["score"].to_numpy(dtype=np.float32),
            "prior_hits": sub["prior_hits"].to_numpy(dtype=np.int32),
            "bbox_xyxy": bbox,
            "feats": fts.astype(np.float32),
        }

    tr = pack(train, tr_feats)
    va = pack(val, va_feats)
    np.savez_compressed(OUT / "train_episodes.npz", **tr)
    np.savez_compressed(OUT / "val_episodes.npz", **va)
    (OUT / "class_split.json").write_text(json.dumps(split, indent=2))
    stats = {
        "train_rows": len(tr["feats"]),
        "val_rows": len(va["feats"]),
        "train_known_rows": int(tr["gt_role"].sum()),
        "val_known_rows": int(va["gt_role"].sum()),
        "train_videos": len(train_videos),
        "val_videos": sorted(val_videos),
        "classes": split,
    }
    (OUT / "episode_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()
    build(args.seed, args.val_frac)


if __name__ == "__main__":
    main()
