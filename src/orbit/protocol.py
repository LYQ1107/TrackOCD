"""Shared IO and protocol helpers for ORBIT."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TRACKOCD = ROOT / "data" / "trackocd_v1"
FEATURE_CACHE = ROOT / "data" / "caches" / "features" / "dinov2"


def load_frame_features(subdir: str) -> dict[str, np.ndarray]:
    out = {}
    cache = FEATURE_CACHE / subdir
    for p in cache.glob("*.json"):
        r = json.loads(p.read_text())
        arr = np.asarray(r["frame_embeddings"], dtype=np.float32)
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        out[r["sample_id"]] = arr
    return out


def load_mean_features(subdir: str) -> dict[str, np.ndarray]:
    out = {}
    cache = FEATURE_CACHE / subdir
    for p in cache.glob("*.json"):
        r = json.loads(p.read_text())
        v = np.asarray(r["mean_embedding"], dtype=np.float32)
        out[r["sample_id"]] = v / (np.linalg.norm(v) + 1e-12)
    return out


def load_train_labels() -> dict[str, int]:
    labels = {}
    with open(TRACKOCD / "pure" / "public" / "train_known_tracks.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                labels[r["sample_id"]] = int(r["category_id"])
    return labels


def load_stream(proto: str, stream: str) -> list[dict]:
    fname = "val_gt_track_stream.jsonl" if stream == "main" else f"val_gt_track_stream_{stream[5:]}.jsonl"
    rows = []
    with open(TRACKOCD / proto / "public" / fname) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_gt(proto: str) -> list[dict]:
    rows = []
    with open(TRACKOCD / proto / "private" / "val_gt_track_labels.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def subset_ids(proto: str, subset: str) -> set[int] | None:
    if subset == "full":
        return None
    p = TRACKOCD / proto / "splits" / f"{subset}_track_ids.json"
    return set(json.loads(p.read_text()))


def meta_classes(kind: str) -> set[int]:
    fname = f"{kind}.csv" if kind.endswith("_classes") else f"{kind}_classes.csv"
    p = ROOT / "outputs" / "orbit" / "splits" / fname
    return {int(r["class_id"]) for r in csv.DictReader(open(p))}
