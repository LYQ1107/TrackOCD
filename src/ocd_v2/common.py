"""Shared IO / evaluation helpers for Architecture 1.5 Stage A."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_predictions, load_private_labels


def load_mean_features(encoder: str, subdir: str):
    cache = PROJECT_ROOT / "data" / "caches" / "features" / encoder / subdir
    feats = {}
    for p in cache.glob("*.json"):
        r = json.loads(p.read_text())
        feats[r["sample_id"]] = np.asarray(r["mean_embedding"], dtype=np.float32)
    return feats


def load_train_known(encoder: str):
    feats = load_mean_features(encoder, "train_known_mean")
    labels = {}
    with open(PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / "train_known_tracks.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["sample_id"] in feats:
                labels[r["sample_id"]] = r["category_id"]
    return feats, labels


def load_train_known_meta():
    """Observable per-track metadata for train-known tracks (no GT category)."""
    meta = {}
    with open(PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / "train_known_tracks.jsonl") as f:
        for line in f:
            r = json.loads(line)
            boxes = r.get("boxes_xyxy") or []
            areas = []
            for b in boxes:
                areas.append((b[2] - b[0]) * (b[3] - b[1]))
            meta[r["sample_id"]] = {
                "num_frames": len(boxes),
                "mean_area": float(np.mean(areas)) if areas else 0.0,
            }
    return meta


def row_meta(row):
    boxes = row.get("boxes_xyxy")
    if boxes:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        mean_area = float(np.mean(areas)) if areas else 0.0
    else:
        areas = row.get("areas")
        mean_area = float(np.mean(areas)) if areas else 0.0
    n_frames = len(row.get("frame_ids", []) or [])
    if n_frames == 0:
        n_frames = int(row.get("num_frames", 1))
    return {"num_frames": n_frames, "mean_area": mean_area}


def load_stream(stream_name: str):
    rows = []
    with open(PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / stream_name) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def proxy_split(labels, seed=1027):
    known_classes = sorted(set(labels.values()))
    rng = random.Random(seed)
    rng.shuffle(known_classes)
    n_half = len(known_classes) // 2
    return set(known_classes[:n_half]), set(known_classes[n_half:])


def build_prototypes(feats, labels, class_ids):
    protos = {}
    sums = {}
    counts = {}
    for sid, cat in labels.items():
        if cat not in class_ids or sid not in feats:
            continue
        sums.setdefault(cat, np.zeros_like(feats[sid]))
        sums[cat] += feats[sid]
        counts[cat] = counts.get(cat, 0) + 1
    for c, s in sums.items():
        v = s / counts[c]
        protos[c] = v / (np.linalg.norm(v) + 1e-12)
    return protos


def subset_keep(rows, subset):
    if subset == "full":
        return np.ones(len(rows), dtype=bool)
    ids = set(
        json.loads(
            (
                PROJECT_ROOT
                / "data"
                / "tao_ow_ocd_v1"
                / "manifests"
                / f"{subset}_track_ids.json"
            ).read_text()
        )
    )
    return np.array([r["sample_id"] in ids for r in rows])


def evaluate_rows(rows, y_true, preds, known_mask, subset, private):
    keep = subset_keep(rows, subset)
    return evaluate_predictions(y_true[keep], preds[keep], known_mask[keep])


def load_val_labels():
    return load_private_labels(PROJECT_ROOT)


def stream_names():
    return [
        "val_gt_track_stream.jsonl",
        "val_gt_track_stream_seed1027.jsonl",
        "val_gt_track_stream_seed1028.jsonl",
        "val_gt_track_stream_seed1029.jsonl",
    ]


def seed_label(stream_name):
    return stream_name.replace("val_gt_track_stream", "main").replace(".jsonl", "").replace("_seed", "_seed")
