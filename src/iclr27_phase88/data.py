"""Phase88 causal feature access.

The old feature stream is reused read-only, but this facade deliberately
keeps loss/evaluator metadata separate from tensors.  Track keys, categories,
videos and row labels are used only to build legal TRAIN events or score
records; model inputs are raw visual vectors, geometry and quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.iclr27_phase19r.data.stream import GEOM, Phase19RData


@dataclass
class TrackArray:
    key: str
    raw: np.ndarray
    geom: np.ndarray
    quality: np.ndarray


def _quality(data: Phase19RData, row_index: int) -> float:
    return float(data._quality_row(row_index))


class FeatureStore:
    """Fold-local, read-only adapter over Phase19RData."""

    def __init__(self, fold: int = 0, data: Phase19RData | Any | None = None):
        self.fold = int(fold)
        self.data = data or Phase19RData(self.fold)

    def keys(self) -> list[str]:
        return sorted(self.data.track_rows)

    def track(self, key: str, max_len: int = 16) -> TrackArray:
        if key not in self.data.track_rows:
            raise KeyError(key)
        indices = self.data.track_rows[key]
        n = min(int(max_len), len(indices))
        idx = np.asarray(indices[:n], dtype=np.int64)
        raw = np.asarray(self.data.raw[idx], dtype=np.float32)
        geom = np.asarray(self.data.geom[idx], dtype=np.float32)
        quality = np.asarray([_quality(self.data, int(i)) for i in idx], dtype=np.float32)
        return TrackArray(key, raw, geom, quality)

    def video(self, key: str) -> int:
        return int(self.data.track_video[key])

    def category(self, key: str) -> int:
        return int(self.data.track_category[key])

    def reliability_prefix(self, key: str, max_len: int = 16) -> int:
        indices = self.data.track_rows[key]
        for pos, row_index in enumerate(indices[:max_len], start=1):
            try:
                if hasattr(self.data, "_assigned"):
                    assigned = int(self.data._assigned[row_index])
                    row_iou = float(self.data._row_iou[row_index])
                else:
                    row = self.data.rows[row_index]
                    assigned = int(row.get("assigned", 0))
                    row_iou = float(row.get("row_iou", 0.0))
                if assigned == 1 and row_iou >= 0.5:
                    return pos
            except (TypeError, ValueError):
                pass
        return min(int(max_len), len(indices))

    def summary(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "rows": len(self.data.rows),
            "tracklets": len(self.data.track_rows),
            "known_count": int(len(self.data.supported_ids)),
            "active_known_count": int(np.asarray(self.data.active_known_mask).sum()),
            "input_fields": ["raw_visual_768", "geometry_15", "quality", "causal_prefix"],
            "loss_only_fields": ["category", "video", "track_key", "row_iou", "assigned"],
            "future_rows_or_tracks": False,
            "ids_or_text_as_model_input": False,
        }


def pad_track(array: TrackArray, length: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = min(int(length), len(array.raw))
    raw = np.zeros((length, 768), dtype=np.float32)
    geom = np.zeros((length, len(GEOM)), dtype=np.float32)
    quality = np.zeros((length,), dtype=np.float32)
    mask = np.zeros((length,), dtype=bool)
    raw[:n] = array.raw[:n]
    geom[:n] = array.geom[:n]
    quality[:n] = array.quality[:n]
    mask[:n] = True
    return raw, geom, quality, mask
