"""Nearest known/anonymous prototype baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


@dataclass
class NearestPrototype:
    tau_known: float = 0.35
    tau_existing: float = 0.55
    known_prototypes: Mapping[int, np.ndarray] = field(default_factory=dict)
    _anonymous: dict[str, tuple[np.ndarray, int]] = field(default_factory=dict, init=False)
    _next_id: int = field(default=0, init=False)

    def reset(self) -> None:
        self._anonymous.clear()
        self._next_id = 0

    def _update(self, token: str, vector: np.ndarray) -> None:
        old, count = self._anonymous[token]
        updated = _normalize((old * count + vector) / (count + 1))
        self._anonymous[token] = (updated, count + 1)

    def step(self, vector: np.ndarray) -> dict[str, str | float]:
        z = _normalize(vector)
        known_scores = {int(category): float(np.dot(z, _normalize(proto))) for category, proto in self.known_prototypes.items()}
        anon_scores = {token: float(np.dot(z, proto)) for token, (proto, _) in self._anonymous.items()}
        best_known = max(known_scores.items(), key=lambda item: item[1], default=(None, -1.0))
        best_anon = max(anon_scores.items(), key=lambda item: item[1], default=(None, -1.0))
        if best_anon[0] is not None and best_anon[1] >= self.tau_existing and best_anon[1] >= best_known[1]:
            token = str(best_anon[0])
            self._update(token, z)
            return {"kind": "EXISTING", "token": token, "score": best_anon[1]}
        if best_known[0] is not None and best_known[1] >= self.tau_known:
            return {"kind": "KNOWN", "token": f"K:{best_known[0]}", "category_id": int(best_known[0]), "score": best_known[1]}
        token = f"A:{self._next_id}"
        self._next_id += 1
        self._anonymous[token] = (z, 1)
        return {"kind": "NEW", "token": token, "score": max(best_known[1], best_anon[1])}
