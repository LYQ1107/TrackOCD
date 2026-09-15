"""The public TrackSample interface.

Ground-truth category fields are evaluator-side annotations.  ``model_view``
and ``full_feature`` intentionally never expose them to a model or method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32, copy=False)


@dataclass
class TrackSample:
    sample_key: str
    video_id: int
    physical_track_id: str
    frame_ids: list[int]
    boxes_xyxy: list[list[float]]
    visual_features: Optional[np.ndarray] = None
    quality: list[float] = field(default_factory=list)
    gt_category_id: Optional[int] = None
    gt_split: Optional[str] = None

    def __post_init__(self) -> None:
        self.frame_ids = [int(x) for x in self.frame_ids]
        self.boxes_xyxy = [[float(v) for v in box] for box in self.boxes_xyxy]
        if len(self.frame_ids) != len(self.boxes_xyxy):
            raise ValueError(f"frame/box length mismatch for {self.sample_key}")
        if any(len(box) != 4 for box in self.boxes_xyxy):
            raise ValueError(f"boxes must be xyxy quadruples for {self.sample_key}")
        if not self.quality:
            self.quality = [1.0] * len(self.frame_ids)
        if len(self.quality) != len(self.frame_ids):
            raise ValueError(f"frame/quality length mismatch for {self.sample_key}")
        self.quality = [max(float(x), 0.0) for x in self.quality]
        if self.visual_features is not None:
            features = np.asarray(self.visual_features, dtype=np.float32)
            if features.ndim != 2 or features.shape[0] != len(self.frame_ids):
                raise ValueError(f"feature shape mismatch for {self.sample_key}: {features.shape}")
            if not np.isfinite(features).all():
                raise ValueError(f"non-finite visual feature for {self.sample_key}")
            self.visual_features = features

    @property
    def length(self) -> int:
        return len(self.frame_ids)

    def _feature(self, limit: int | None) -> np.ndarray:
        if self.visual_features is None:
            raise ValueError(f"visual features are unavailable for {self.sample_key}")
        end = self.length if limit is None else int(limit)
        if end < 1 or end > self.length:
            raise ValueError(f"prefix {end} outside 1..{self.length} for {self.sample_key}")
        values = self.visual_features[:end]
        weights = np.asarray(self.quality[:end], dtype=np.float32)
        if float(weights.sum()) <= 1e-12:
            weights = np.ones(end, dtype=np.float32)
        return _unit((values * weights[:, None]).sum(axis=0) / weights.sum())

    def full_feature(self) -> np.ndarray:
        return self._feature(None)

    def prefix_feature(self, k: int) -> np.ndarray:
        return self._feature(k)

    def model_view(self) -> dict[str, Any]:
        """Return only fields permitted as method input."""

        return {
            "sample_key": self.sample_key,
            "video_id": self.video_id,
            "physical_track_id": self.physical_track_id,
            "frame_ids": list(self.frame_ids),
            "boxes_xyxy": [list(box) for box in self.boxes_xyxy],
            "visual_features": self.visual_features,
            "quality": list(self.quality),
        }

    def evaluator_view(self) -> dict[str, Any]:
        result = self.model_view()
        result.update({"gt_category_id": self.gt_category_id, "gt_split": self.gt_split})
        return result
