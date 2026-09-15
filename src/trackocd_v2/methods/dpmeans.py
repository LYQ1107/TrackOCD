"""Online DP-Means anonymous clustering baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


@dataclass
class OnlineDPMeans:
    lambda_distance: float = 0.45
    tau_known: float = 0.35
    known_prototypes: Mapping[int, np.ndarray] = field(default_factory=dict)
    _centroids: dict[str, tuple[np.ndarray, int]] = field(default_factory=dict, init=False)
    _next_id: int = field(default=0, init=False)

    def reset(self) -> None:
        self._centroids.clear()
        self._next_id = 0

    def step(self, vector: np.ndarray) -> dict[str, str | float]:
        z = _normalize(vector)
        existing = [(token, float(1.0 - np.dot(z, centroid)), centroid) for token, (centroid, _) in self._centroids.items()]
        if existing:
            token, distance, _ = min(existing, key=lambda item: item[1])
            if distance <= self.lambda_distance:
                centroid, count = self._centroids[token]
                self._centroids[token] = (_normalize((centroid * count + z) / (count + 1)), count + 1)
                return {"kind": "EXISTING", "token": token, "distance": distance}
        known = [(int(category), float(1.0 - np.dot(z, _normalize(proto)))) for category, proto in self.known_prototypes.items()]
        if known:
            category, distance = min(known, key=lambda item: item[1])
            if distance <= self.tau_known:
                return {"kind": "KNOWN", "token": f"K:{category}", "category_id": category, "distance": distance}
        token = f"A:{self._next_id}"
        self._next_id += 1
        self._centroids[token] = (z, 1)
        return {"kind": "NEW", "token": token, "distance": min([x[1] for x in existing] + [1.0])}
