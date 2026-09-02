from __future__ import annotations

import torch
from torch import nn


class SelectiveRelationRouter(nn.Module):
    """Predict HELP/HARM/NEUTRAL from causal raw/relation diagnostics.

    The class label is TRAIN-only metadata; the input is a small vector of
    raw-bank uncertainty and frozen relation statistics.  Inference selects
    the relation scores only when HELP is the argmax; otherwise it returns the
    exact raw scores.
    """

    input_dim = 14
    classes = ("HELP", "HARM", "NEUTRAL")

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(self.input_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 3))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"router features must be [N,{self.input_dim}], got {tuple(features.shape)}")
        return self.net(features)
