"""Frozen-frame pair matching and symmetric relation features."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def normalize_frames(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 768:
        raise ValueError(f"expected [T,768], got {value.shape}")
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)


def hungarian_match(query_frames: np.ndarray, candidate_frames: np.ndarray) -> dict[str, Any]:
    """Return detached CPU one-to-one matches and their frozen similarities."""
    q = normalize_frames(query_frames)
    c = normalize_frames(candidate_frames)
    sim = q @ c.T
    if not sim.size:
        return {"q_indices": [], "c_indices": [], "similarities": [], "matrix_shape": [int(q.shape[0]), int(c.shape[0])]}
    qi, ci = linear_sum_assignment(-sim)
    return {
        "q_indices": [int(x) for x in qi.tolist()],
        "c_indices": [int(x) for x in ci.tolist()],
        "similarities": [float(x) for x in sim[qi, ci].tolist()],
        "matrix_shape": [int(q.shape[0]), int(c.shape[0])],
    }


def pair_relation_features(query_frames: np.ndarray, candidate_frames: np.ndarray, match: dict[str, Any]) -> np.ndarray:
    """Symmetric ``concat(abs(q-c), q*c)`` token features (1536-D)."""
    q = normalize_frames(query_frames)
    c = normalize_frames(candidate_frames)
    qi = np.asarray(match.get("q_indices", []), dtype=np.int64)
    ci = np.asarray(match.get("c_indices", []), dtype=np.int64)
    if len(qi) == 0:
        return np.zeros((0, 1536), dtype=np.float32)
    qa, ca = q[qi], c[ci]
    return np.concatenate([np.abs(qa - ca), qa * ca], axis=1).astype(np.float32, copy=False)


def _adjacent_consistency(x: np.ndarray) -> float:
    value = normalize_frames(x)
    if len(value) < 2:
        return 1.0 if len(value) else 0.0
    return float(np.mean(np.sum(value[:-1] * value[1:], axis=1)))


def relation_summary(query_frames: np.ndarray, candidate_frames: np.ndarray, match: dict[str, Any], raw_cosine: float) -> np.ndarray:
    """Return the registered 13 scalar summary features in fixed order."""
    q = normalize_frames(query_frames); c = normalize_frames(candidate_frames)
    sim = q @ c.T
    matched = np.asarray(match.get("similarities", []), dtype=np.float32)
    row_best = np.max(sim, axis=1) if sim.size else np.asarray([], dtype=np.float32)
    col_best = np.max(sim, axis=0) if sim.size else np.asarray([], dtype=np.float32)
    qi = np.asarray(match.get("q_indices", []), dtype=np.int64)
    ci = np.asarray(match.get("c_indices", []), dtype=np.int64)
    mutual = 0.0
    if len(qi):
        row_arg = np.argmax(sim, axis=1); col_arg = np.argmax(sim, axis=0)
        mutual = float(np.mean([(row_arg[int(qi[k])] == int(ci[k]) and col_arg[int(ci[k])] == int(qi[k])) for k in range(len(qi))]))
    coverage = float(len(qi) / max(min(len(q), len(c)), 1))
    values = [
        float(raw_cosine),
        float(matched.mean()) if len(matched) else 0.0,
        float(matched.std()) if len(matched) else 0.0,
        float(matched.min()) if len(matched) else 0.0,
        float(matched.max()) if len(matched) else 0.0,
        float(row_best.mean()) if len(row_best) else 0.0,
        float(col_best.mean()) if len(col_best) else 0.0,
        mutual,
        coverage,
        float(min(len(q), 16) / 16.0),
        float(min(len(c), 16) / 16.0),
        _adjacent_consistency(q),
        _adjacent_consistency(c),
    ]
    return np.asarray(values, dtype=np.float32)

