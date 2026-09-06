"""Explicit causal support evidence interface."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SupportEvidence:
    available: bool
    raw_best: float = 0.0
    raw_second: float = 0.0
    raw_margin: float = 0.0
    candidate_count_norm: float = 0.0
    source_length_norm: float = 0.0
    source_variance: float = 0.0
    observation_quality: float = 0.0
    history_consistency: float = 0.0

    def tensor(self, device: torch.device | str = "cpu") -> torch.Tensor:
        if not self.available:
            return torch.zeros((8,), dtype=torch.float32, device=device)
        return torch.tensor([self.raw_best, self.raw_second, self.raw_margin, self.candidate_count_norm, self.source_length_norm, self.source_variance, self.observation_quality, self.history_consistency], dtype=torch.float32, device=device)
