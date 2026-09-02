"""Exact frame-set pairwise correspondence with deterministic Hungarian score."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .protocol import l2_normalize


def pairwise_similarity(query_frames: np.ndarray, candidate_frames: np.ndarray) -> np.ndarray:
    q = l2_normalize(np.asarray(query_frames, dtype=np.float32))
    c = l2_normalize(np.asarray(candidate_frames, dtype=np.float32))
    if q.ndim != 2 or c.ndim != 2 or q.shape[1] != 768 or c.shape[1] != 768:
        raise ValueError(f"expected [T,768] arrays, got {q.shape} and {c.shape}")
    return q @ c.T


def _score_matrix(s: np.ndarray) -> float:
    """Exact assignment score with small-size fast paths.

    The fast paths are algebraically identical to a rectangular Hungarian
    assignment and only avoid SciPy call overhead for one/two-frame prefixes.
    """
    if s.size == 0 or not s.shape[0] or not s.shape[1]:
        return float("-inf")
    q, c = s.shape
    if q > c:
        return _score_matrix(s.T)
    if q == 1:
        return float(np.max(s))
    if q == 2:
        best = -np.inf
        for j in range(c):
            rest = np.max(np.delete(s[1], j)) if c > 1 else -np.inf
            best = max(best, float(s[0, j] + rest))
        return float(best / 2.0)
    ri, ci = linear_sum_assignment(-s)
    return float(s[ri, ci].mean()) if len(ri) else float("-inf")


def hungarian_score(query_frames: np.ndarray, candidate_frames: np.ndarray, *, diagnostic: bool = False) -> float | tuple[float, dict[str, Any]]:
    """Return mean matched cosine; assignment is non-differentiable by design."""
    s = pairwise_similarity(query_frames, candidate_frames)
    if s.size == 0:
        score = float("-inf")
        pairs: tuple[np.ndarray, np.ndarray] = (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64))
    else:
        ri, ci = linear_sum_assignment(-s)
        score = float(s[ri, ci].mean()) if len(ri) else float("-inf")
        pairs = (ri, ci)
    if not diagnostic:
        return score
    ri, ci = pairs
    matched = s[ri, ci] if len(ri) else np.asarray([], dtype=np.float32)
    return score, {
        "query_length": int(s.shape[0]), "candidate_length": int(s.shape[1]),
        "matched_pairs": int(len(ri)),
        "mean_similarity": None if not len(matched) else float(matched.mean()),
        "min_similarity": None if not len(matched) else float(matched.min()),
        "max_similarity": None if not len(matched) else float(matched.max()),
        "row_ind": [int(x) for x in ri.tolist()], "col_ind": [int(x) for x in ci.tolist()],
    }


def fast_hungarian_score(query_frames: np.ndarray, candidate_frames: np.ndarray) -> float:
    """Same primary score as :func:`hungarian_score`, with exact small paths."""
    return _score_matrix(pairwise_similarity(query_frames, candidate_frames))
