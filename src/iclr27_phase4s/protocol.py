"""Phase 4S shared protocol helpers.

Paths, frozen splits, and loaders for:
  - the legal pseudo-novel training universe (48 train-supported known cats),
  - the frozen dev physical frontends (Q1 / Q2-alpha0.1),
  - TrackOCD-v1.0 private GT labels (evaluation only).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TRACKOCD = ROOT / "data" / "trackocd_v1"
FEATURES = ROOT / "data" / "caches" / "features" / "dinov2"
TAO_VAL_ANN = ROOT / "data" / "raw" / "tao" / "annotations" / "validation.json"
DEV_GT_JSON = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset" / "validation_20.json"
DEV_VIDEOS_CSV = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "selected_20_videos.csv"
Q1_DEV = ROOT / "outputs" / "iclr27_phase4q" / "q1_long" / "proposals_dev.csv"
Q2_DEV = ROOT / "outputs" / "iclr27_phase4r" / "q2_alpha" / "a010" / "proposals_dev.csv"
META_SPLIT = ROOT / "outputs" / "orbit" / "splits"


def known_ids() -> set[int]:
    return set(json.loads((TRACKOCD / "pure" / "splits" / "supported_known_ids.json").read_text()))


def meta_split_classes() -> tuple[set[int], set[int]]:
    def read(name: str) -> set[int]:
        with open(META_SPLIT / name) as f:
            return {int(r["class_id"]) for r in csv.DictReader(f)}
    return read("meta_train_classes.csv"), read("meta_dev_classes.csv")


def load_train_tracks() -> list[dict]:
    rows = []
    with open(TRACKOCD / "pure" / "public" / "train_known_tracks.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_frame_features(sample_ids: list[str] | None = None) -> dict[str, np.ndarray]:
    """Per-frame L2-normalized DINOv2 (768-d) features for train-known tracks."""
    out: dict[str, np.ndarray] = {}
    for sid in (sample_ids or [p.stem for p in (FEATURES / "train_known_mean").glob("*.json")]):
        p = FEATURES / "train_known_mean" / f"{sid}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        arr = np.asarray(r["frame_embeddings"], dtype=np.float32)
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        out[sid] = arr
    return out


def load_mean_features(sample_ids: list[str] | None = None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for sid in (sample_ids or [p.stem for p in (FEATURES / "train_known_mean").glob("*.json")]):
        p = FEATURES / "train_known_mean" / f"{sid}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        v = np.asarray(r["mean_embedding"], dtype=np.float32)
        out[sid] = v / (np.linalg.norm(v) + 1e-12)
    return out


def load_dev_videos() -> list[int]:
    with open(DEV_VIDEOS_CSV) as f:
        return [int(r["video_id"]) for r in csv.DictReader(f)]


def load_gt_tracks_dev() -> tuple[list[dict], dict[str, dict]]:
    """GT physical tracks restricted to the frozen 20-video dev set.

    Returns (stream_rows, label_by_sample_id). Label rows carry
    ground_truth_category_id and protocol_role (evaluation only).
    """
    dev_videos = set(load_dev_videos())
    stream = []
    with open(TRACKOCD / "pure" / "public" / "val_gt_track_stream.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if int(r["video_id"]) in dev_videos:
                    stream.append(r)
    labels = {}
    with open(TRACKOCD / "pure" / "private" / "val_gt_track_labels.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                labels[r["sample_id"]] = r
    stream = [r for r in stream if r["sample_id"] in labels]
    return stream, labels


def load_proposals(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r = dict(r)
            r["video_id"] = int(r["video_id"])
            r["frame_id"] = int(r["frame_id"])
            r["track_id"] = int(r["track_id"])
            r["score"] = float(r["score"])
            r["prior_hits"] = int(r.get("prior_hits") or 0)
            r["gt_iou"] = float(r.get("gt_iou") or 0.0)
            r["gt_role"] = r.get("gt_role") or "fp"
            r["gt_category_id"] = int(r.get("gt_category_id") or -1)
            rows.append(r)
    return rows


def temporal_iou(gt_boxes: dict[int, list[float]], pred_boxes: dict[int, list[float]]) -> float:
    """IoU over the union of frames, weighted by the number of shared frames."""
    shared = [f for f in pred_boxes if f in gt_boxes]
    if not shared:
        return 0.0
    total = 0.0
    for f in shared:
        total += box_iou(gt_boxes[f], pred_boxes[f])
    return total / max(len(shared), 1)


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def group_tracks(rows: list[dict]) -> dict[tuple[int, int], list[dict]]:
    tracks = defaultdict(list)
    for r in rows:
        tracks[(r["video_id"], r["track_id"])].append(r)
    for k in tracks:
        tracks[k].sort(key=lambda r: (r["frame_id"], int(r.get("proposal_local_id") or 0)))
    return tracks


def taq_image_path(image_id: int) -> str | None:
    d = json.loads(TAO_VAL_ANN.read_text())
    for im in d["images"]:
        if int(im["id"]) == image_id:
            return im["file_name"]
    return None
