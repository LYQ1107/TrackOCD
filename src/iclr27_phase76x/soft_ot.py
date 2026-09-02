from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)


def uniform_sinkhorn_score(query_frames: np.ndarray, candidate_frames: np.ndarray, *, temperature: float = 0.07, iterations: int = 50) -> float:
    """Causal symmetric soft-OT similarity with uniform frame marginals.

    The query is the requested prefix and the candidate is its available
    prefix.  Uniform marginals prevent track length from becoming a score
    shortcut.  No labels, IDs or future frames enter this operation.
    """
    q = normalize(query_frames); c = normalize(candidate_frames)
    if not len(q) or not len(c): return 0.0
    sim = np.asarray(q @ c.T, dtype=np.float64)
    tau = max(float(temperature), 1e-4)
    logits = (sim - np.max(sim)) / tau
    kernel = np.exp(np.clip(logits, -60.0, 0.0))
    a = np.full(len(q), 1.0 / len(q), dtype=np.float64)
    b = np.full(len(c), 1.0 / len(c), dtype=np.float64)
    u = np.ones_like(a); v = np.ones_like(b)
    for _ in range(int(iterations)):
        u = a / np.maximum(kernel @ v, 1e-12)
        v = b / np.maximum(kernel.T @ u, 1e-12)
    plan = u[:, None] * kernel * v[None, :]
    return float(np.sum(plan * sim) / np.maximum(np.sum(plan), 1e-12))


def anchored_ot_score(query_frames: np.ndarray, candidate_frames: np.ndarray, raw_score: float, *, alpha: float = 0.5) -> float:
    """Fixed raw-preserving convex anchor used by the registered route."""
    ot = uniform_sinkhorn_score(query_frames, candidate_frames)
    weight = min(max(float(alpha), 0.0), 1.0)
    return float(weight * float(raw_score) + (1.0 - weight) * ot)
