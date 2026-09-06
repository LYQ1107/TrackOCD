"""Causal feature access for Phase87.

Track/category labels are read only for TRAIN event construction and target
losses.  The tensors returned here contain visual features, geometry, quality
and causal age only.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase75d.protocol import FrozenTrackTable, load_frozen_tracks


GEOM_FIELDS = (
    "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm",
    "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log",
    "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm",
    "causal_prefix_age_norm", "causal_prefix_count", "causal_box_stability_iou",
)


@dataclass
class TrackArray:
    key: str
    raw: np.ndarray
    geom: np.ndarray
    quality: np.ndarray


class FeatureStore:
    def __init__(self, table: FrozenTrackTable | None = None) -> None:
        self.table = table or load_frozen_tracks()
        self._cache: dict[str, TrackArray] = {}

    def keys(self) -> list[str]:
        return sorted(self.table.sequences)

    def track(self, key: str, max_len: int = 16) -> TrackArray:
        if key in self._cache:
            value = self._cache[key]
            return TrackArray(key, value.raw[:max_len], value.geom[:max_len], value.quality[:max_len])
        sequence = self.table.sequences[key]
        indices = list(sequence.row_indices)
        raw = self.table.features[np.asarray(indices, dtype=np.int64)].astype(np.float32)
        geom_rows: list[list[float]] = []
        quality: list[float] = []
        for index in indices:
            row = self.table.rows[index]
            geom_rows.append([float(row.get(field, 0.0) or 0.0) for field in GEOM_FIELDS])
            quality.append(float(row.get("score", 0.0) or 0.0))
        value = TrackArray(key, raw, np.asarray(geom_rows, np.float32), np.asarray(quality, np.float32))
        self._cache[key] = value
        return TrackArray(key, value.raw[:max_len], value.geom[:max_len], value.quality[:max_len])


def reliability_prefix(table: FrozenTrackTable, key: str, max_len: int = 16) -> int:
    """TRAIN-only label-side reliable prefix; never returned as model input."""
    sequence = table.sequences[key]
    for position, index in enumerate(sequence.row_indices[:max_len], start=1):
        row = table.rows[index]
        try:
            reliable = int(row.get("assigned", 0)) == 1 and float(row.get("row_iou", 0.0)) >= 0.5
        except (TypeError, ValueError):
            reliable = False
        if reliable:
            return position
    return min(max_len, len(sequence.row_indices))


def pad_track(array: TrackArray, length: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = min(length, len(array.raw))
    raw = np.zeros((length, 768), np.float32)
    geom = np.zeros((length, 15), np.float32)
    quality = np.zeros((length,), np.float32)
    mask = np.zeros((length,), dtype=bool)
    raw[:n] = array.raw[:n]
    geom[:n] = array.geom[:n]
    quality[:n] = array.quality[:n]
    mask[:n] = True
    return raw, geom, quality, mask
