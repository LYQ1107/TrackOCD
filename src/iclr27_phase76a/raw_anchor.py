"""Immutable raw cosine comparator used by Phase76A.

The global scorer has no trainable state.  It intentionally mirrors the
Phase75D ``FrozenTrackTable.raw_vector`` operation: normalize each frame,
average the causal prefix, then normalize the resulting track vector.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    value = np.asarray(x, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=axis, keepdims=True), 1e-8)


def raw_mean_vector(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 768:
        raise ValueError(f"expected [T,768], got {arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros(768, dtype=np.float32)
    # ``FrozenTrackTable.get_frame_sequence`` already returns rows normalized
    # in float32.  Re-normalizing those rows changes the dot product by a few
    # ulps and breaks the registered Phase75D <=1e-7 parity check.  For raw
    # inputs (e.g. a standalone caller) retain the normalizing behavior.
    norms = np.linalg.norm(arr, axis=1)
    frame = arr if np.all(np.abs(norms - 1.0) <= 1e-5) else _normalize(arr)
    return _normalize(frame.mean(axis=0, keepdims=True))[0]


def raw_mean_cosine(query_frames: np.ndarray, candidate_frames: np.ndarray) -> float:
    q = raw_mean_vector(query_frames)
    c = raw_mean_vector(candidate_frames)
    return float(np.dot(q, c))


class RawAnchorScorer:
    """Stateless global comparator; no parameters and no category/ID inputs."""

    def score(self, query_frames: np.ndarray, candidate_frames: np.ndarray) -> float:
        return raw_mean_cosine(query_frames, candidate_frames)

    def score_many(self, query_frames: np.ndarray, candidates: Iterable[np.ndarray]) -> list[float]:
        return [self.score(query_frames, c) for c in candidates]
