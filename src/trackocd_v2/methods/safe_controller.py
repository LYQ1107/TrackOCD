"""A conservative causal probability hierarchy for TrackOCD v2.

The controller keeps physical tracking and semantic memory separate.  It
receives only a normalized visual track representation and optional causal
quality/length metadata.  The three probability layers are explicit:

    WAIT/READY -> KNOWN/OPEN -> KNOWN_k or EXISTING_j/NEW

No category labels, text embeddings, or future observations are required at
inference time.  Thresholds are constructor parameters so Val selection can
freeze them before any Test evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"expected a vector, got shape {vector.shape}")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or not np.isfinite(norm):
        return np.zeros_like(vector)
    return vector / norm


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    shifted = values - np.max(values)
    exp = np.exp(np.clip(shifted, -60.0, 60.0))
    return (exp / max(float(exp.sum()), 1e-12)).astype(np.float32)


@dataclass
class _AnonymousState:
    prototype: np.ndarray
    count: int = 1


@dataclass
class SafePersistentController:
    """Causal, conservative semantic state machine.

    ``tau_*`` are probability thresholds, not cosine thresholds.  The
    temperature and maturity parameters are fixed by the Val selection
    artifact.  A state is updated only after an EXISTING decision, so a low
    confidence observation cannot silently alter persistent memory.
    """

    known_prototypes: Mapping[int, np.ndarray] = field(default_factory=dict)
    temperature: float = 0.08
    maturity_temperature: float = 1.0
    tau_known: float = 0.70
    tau_existing: float = 0.70
    tau_new: float = 0.45
    min_observations: int = 2
    _anonymous: dict[str, _AnonymousState] = field(default_factory=dict, init=False)
    _next_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.known_prototypes = {
            int(category): _unit(np.asarray(vector, dtype=np.float32))
            for category, vector in self.known_prototypes.items()
        }
        if self.temperature <= 0 or self.maturity_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.min_observations < 1:
            raise ValueError("min_observations must be positive")

    def reset(self) -> None:
        self._anonymous.clear()
        self._next_id = 0

    def _probabilities(self, vector: np.ndarray, observations: int, quality: float) -> tuple[dict[int, float], dict[str, float], float, float]:
        z = _unit(vector)
        if not np.any(z):
            return {}, {}, 0.0, 0.0
        # A track is not considered mature until the registered minimum number
        # of causal observations has arrived.  Quality is a physical-stream
        # scalar, never a GT/evaluator field.
        quality = float(np.clip(quality, 0.0, 1.0))
        maturity_logit = (float(observations) - float(self.min_observations) + 0.5) / self.maturity_temperature
        p_ready = float(1.0 / (1.0 + np.exp(-maturity_logit)) * (0.5 + 0.5 * quality))
        p_wait = 1.0 - p_ready

        known_scores = {category: float(np.dot(z, prototype)) for category, prototype in self.known_prototypes.items()}
        anon_scores = {token: float(np.dot(z, state.prototype)) for token, state in self._anonymous.items()}
        best_known = max(known_scores.values(), default=-1.0)
        best_anon = max(anon_scores.values(), default=-1.0)
        # When no anonymous state exists, ``-1`` is not evidence against the
        # OPEN branch: it only means NEW is the available open-world action.
        # Use the complement of the best known similarity as that branch's
        # prior, while allowing an existing state to override it when its
        # similarity is strong.
        open_score = max(best_anon, 1.0 - best_known)
        router = _softmax(np.asarray([best_known, open_score], dtype=np.float32) / self.temperature)
        p_known = float(router[0]) if len(router) else 0.0
        p_open = float(router[1]) if len(router) else 0.0
        # Return the branch-local probabilities.  The caller multiplies them
        # by p_ready/p_known/p_open, which prevents an OPEN decision from
        # bypassing the maturity gate.
        return known_scores, anon_scores, p_ready, p_wait

    def step(self, vector: np.ndarray, *, observations: int = 16, quality: float = 1.0) -> dict[str, Any]:
        z = _unit(vector)
        known_scores, anon_scores, p_ready, p_wait = self._probabilities(z, observations, quality)
        if not np.any(z):
            return {"kind": "DEFER", "token": None, "probability": 0.0, "reason": "zero_or_nonfinite_feature"}
        best_known_category, best_known_score = max(known_scores.items(), key=lambda item: item[1], default=(None, -1.0))
        best_anon_token, best_anon_score = max(anon_scores.items(), key=lambda item: item[1], default=(None, -1.0))
        open_router_score = max(best_anon_score, 1.0 - best_known_score)
        router = _softmax(np.asarray([best_known_score, open_router_score], dtype=np.float32) / self.temperature)
        p_known_branch = float(router[0]) if len(router) else 0.0
        p_open_branch = float(router[1]) if len(router) else 0.0
        if p_wait >= p_ready or p_ready < self.tau_new:
            return {
                "kind": "DEFER", "token": None,
                "probability": p_wait,
                "maturity": {"wait": p_wait, "ready": p_ready},
                "reason": "evidence_not_mature",
            }

        if best_known_category is not None:
            known_local = _softmax(np.asarray(list(known_scores.values()), dtype=np.float32) / self.temperature)
            category_index = list(known_scores).index(best_known_category)
            p_known_category = float(known_local[category_index])
        else:
            p_known_category = 0.0
        p_known = p_ready * p_known_branch * p_known_category
        if best_known_category is not None and p_known >= self.tau_known:
            return {
                "kind": "KNOWN", "token": f"K:{int(best_known_category)}", "category_id": int(best_known_category),
                "probability": p_known, "maturity": {"wait": p_wait, "ready": p_ready},
                "router": {"known": p_known_branch, "open": p_open_branch}, "score": best_known_score,
            }

        open_items = list(anon_scores.items()) + [("__NEW__", 0.0)]
        open_local = _softmax(np.asarray([score for _token, score in open_items], dtype=np.float32) / self.temperature)
        if best_anon_token is not None:
            anon_index = [token for token, _score in open_items].index(best_anon_token)
            p_existing = p_ready * p_open_branch * float(open_local[anon_index])
        else:
            p_existing = 0.0
        new_index = [token for token, _score in open_items].index("__NEW__")
        p_new = p_ready * p_open_branch * float(open_local[new_index])
        if best_anon_token is not None and p_existing >= self.tau_existing:
            state = self._anonymous[best_anon_token]
            state.prototype = _unit((state.prototype * state.count + z) / float(state.count + 1))
            state.count += 1
            return {
                "kind": "EXISTING", "token": best_anon_token, "probability": p_existing,
                "maturity": {"wait": p_wait, "ready": p_ready},
                "router": {"known": p_known_branch, "open": p_open_branch}, "score": best_anon_score,
            }
        if p_new >= self.tau_new:
            token = f"A:{self._next_id}"
            self._next_id += 1
            self._anonymous[token] = _AnonymousState(prototype=z.copy())
            return {
                "kind": "NEW", "token": token, "probability": p_new,
                "maturity": {"wait": p_wait, "ready": p_ready},
                "router": {"known": p_known_branch, "open": p_open_branch}, "score": p_new,
            }
        return {
            "kind": "DEFER", "token": None, "probability": max(p_known, p_existing, p_new),
            "maturity": {"wait": p_wait, "ready": p_ready},
            "router": {"known": p_known_branch, "open": p_open_branch},
            "reason": "no_safe_commit",
        }
