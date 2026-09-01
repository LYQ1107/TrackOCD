"""Architecture B: amortized assign-or-create head.

The create logit is learned from the current trajectory representation plus
physical-stream reliability features (objectness score, prior hits, track
age) and the best existing-state similarity.  Existing-state matching is a
temperature-scaled attention over the dynamic prototype memory.  This is a
genuinely different paradigm from Architecture A's Gaussian posterior:
the birth decision is amortized and can use physical evidence, while the
state memory remains online and dynamic.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CreateHead(nn.Module):
    def __init__(self, dim=128, hidden=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim + 5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h, phys, best_sim):
        """h: (dim,), phys: (4,), best_sim: scalar tensor."""
        x = torch.cat([h, phys, best_sim.reshape(1)], dim=0)
        return self.fc(x).reshape(())


def phys_features(row, age, device):
    score = float(row.get("score") or 0.0)
    prior = min(float(row.get("prior_hits") or 0.0), 20.0) / 20.0
    age_n = min(float(age), 50.0) / 50.0
    return torch.tensor(
        [score, prior, age_n, 1.0 - score], device=device)
