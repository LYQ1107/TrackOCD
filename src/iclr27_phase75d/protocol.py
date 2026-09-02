"""Frozen Phase75D protocol facade.

The feature/row artifacts are read-only Phase26/15S inputs.  Track keys and
labels are bookkeeping for split construction and scoring only; scorer
functions receive arrays, never identifiers or category values.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Phase26 is the frozen public facade.  Its import also re-exports the
# dependency-heavy ranker helpers, so use the identical Phase23 primitives
# directly here when the lightweight test environment has no torch installed.
from src.iclr27_phase23.protocol import CSV_PATH, FEAT_PATH, by_track, load_aligned_features, order_key


PREFIXES = (1, 2, 4, 8, 16)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def l2_normalize(value: np.ndarray, axis: int = -1) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.maximum(denom, 1e-8)


@dataclass(frozen=True)
class TrackSequence:
    key: str
    video_id: int
    row_indices: tuple[int, ...]


@dataclass
class FrozenTrackTable:
    rows: list[dict[str, str]]
    sequences: dict[str, TrackSequence]
    metadata: dict[str, dict[str, Any]]
    features: np.ndarray
    alignment: dict[str, Any]
    csv_sha256: str
    feature_sha256: str

    def get_frame_sequence(self, track_key: str, prefix: int | None = None) -> np.ndarray:
        """Return only the causal prefix, never suffix statistics."""
        seq = self.sequences[track_key]
        n = len(seq.row_indices) if prefix is None else min(max(int(prefix), 0), len(seq.row_indices))
        idx = np.asarray(seq.row_indices[:n], dtype=np.int64)
        if len(idx) == 0:
            return np.zeros((0, self.features.shape[1]), dtype=np.float32)
        return l2_normalize(self.features[idx])

    def raw_vector(self, track_key: str, prefix: int | None = None) -> np.ndarray:
        seq = self.get_frame_sequence(track_key, prefix)
        if not len(seq):
            return np.zeros(self.features.shape[1], dtype=np.float32)
        return l2_normalize(seq.mean(axis=0))


def load_frozen_tracks() -> FrozenTrackTable:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cls, roi, alignment = load_aligned_features(rows)
    fused = l2_normalize((0.8 * cls.astype(np.float32) + 0.2 * roi.astype(np.float32)).astype(np.float32))
    tracks = by_track(rows)
    sequences: dict[str, TrackSequence] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for key, indices in tracks.items():
        ordered = tuple(sorted((int(i) for i in indices), key=lambda i: order_key(rows[i])))
        if not ordered:
            continue
        labelled = [rows[i] for i in ordered if rows[i].get("gt_category_id_common") not in {"", "-1", "None", None}]
        if not labelled:
            continue
        category = int(labelled[-1]["gt_category_id_common"])
        if category < 0:
            continue
        video = int(rows[ordered[-1]]["video_id"])
        sequences[str(key)] = TrackSequence(key=str(key), video_id=video, row_indices=ordered)
        metadata[str(key)] = {"category": category, "video": video, "length": len(ordered)}
    if fused.ndim != 2 or fused.shape[1] != 768:
        raise RuntimeError(f"expected aligned fused features [N,768], got {fused.shape}")
    return FrozenTrackTable(
        rows=rows,
        sequences=sequences,
        metadata=metadata,
        features=fused,
        alignment=alignment,
        csv_sha256=sha256(CSV_PATH),
        feature_sha256=sha256(FEAT_PATH),
    )
