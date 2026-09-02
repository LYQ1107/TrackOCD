"""Causal, frozen-feature Grounded Correspondence primitives.

The ICML 2026 Grounded Correspondence recipe replaces a learned temporal
predictor with deterministic correspondence on frozen self-supervised visual
features.  This adapter keeps that property: it has no trainable parameters,
never receives category/track identifiers, and only aggregates the prefix
presented to it.  A track vector is a consistency-weighted temporal medoid;
the same sequence representation is used at every causal prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def l2_normalize(value: np.ndarray, axis: int = -1) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.maximum(denom, 1e-8)


@dataclass(frozen=True)
class GroundedConfig:
    """Fixed, pre-registered representation constants."""

    temperature: float = 0.20
    min_frames: int = 1
    output_dim: int = 768


class GroundedCorrespondence:
    """Parameter-free causal track representation.

    ``encode`` consumes only a feature prefix.  Per-frame weights are based on
    agreement with the other frames in that prefix, and the output is the
    normalized weighted feature average.  This is deliberately not a learned
    GRU/MLP residual and cannot overwrite the frozen raw feature cache.
    """

    def __init__(self, config: GroundedConfig | None = None) -> None:
        self.config = config or GroundedConfig()

    def encode(self, sequence: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(sequence, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.config.output_dim:
            raise ValueError(f"expected [T,{self.config.output_dim}] sequence, got {x.shape}")
        if mask is None:
            valid = np.ones(x.shape[0], dtype=bool)
        else:
            valid = np.asarray(mask, dtype=bool)
            if valid.shape != (x.shape[0],):
                raise ValueError(f"mask shape {valid.shape} does not match sequence {x.shape}")
        x = x[valid]
        if x.shape[0] < self.config.min_frames:
            return np.zeros(self.config.output_dim, dtype=np.float32)
        x = l2_normalize(x)
        # Frozen-feature local consistency, equivalent to choosing grounded
        # salient/medoid evidence without learning a temporal transition.
        similarity = np.clip(x @ x.T, -1.0, 1.0)
        agreement = similarity.mean(axis=1)
        weights = np.exp((agreement - agreement.max()) / max(self.config.temperature, 1e-6))
        weights /= max(float(weights.sum()), 1e-8)
        return l2_normalize((weights[:, None] * x).sum(axis=0))

    def encode_prefix(self, sequence: np.ndarray, prefix: int) -> np.ndarray:
        n = min(max(int(prefix), 0), int(sequence.shape[0]))
        return self.encode(sequence[:n])

    @staticmethod
    def metadata() -> dict[str, Any]:
        return {
            "method": "Grounded Correspondence",
            "learnable_parameters": 0,
            "input": ["frozen_dinov2_cls_roi", "causal_track_prefix"],
            "output": "l2_normalized_768d_track_vector",
            "temporal_operation": "frozen-feature consistency-weighted set/medoid aggregation",
            "forbidden_inputs": [
                "category", "category_text", "physical_id", "semantic_id",
                "future_frame", "future_track", "held_gt", "controller_action",
            ],
            "causal": True,
        }


def hungarian_match_score(left: np.ndarray, right: np.ndarray) -> float:
    """Return deterministic one-to-one frame correspondence score.

    This diagnostic is intentionally separate from the vector output and is
    only used on already materialized prefixes.  SciPy is imported lazily so
    the main vector route remains dependency-light.
    """
    from scipy.optimize import linear_sum_assignment

    a = l2_normalize(np.asarray(left, dtype=np.float32))
    b = l2_normalize(np.asarray(right, dtype=np.float32))
    if a.ndim != 2 or b.ndim != 2 or not len(a) or not len(b):
        return 0.0
    sim = a @ b.T
    ri, ci = linear_sum_assignment(-sim)
    if len(ri) == 0:
        return 0.0
    return float(sim[ri, ci].mean())

