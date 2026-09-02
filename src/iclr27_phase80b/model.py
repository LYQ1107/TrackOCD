"""Small causal candidate-set evidence scorer for Family B.

The model consumes only a sequence of visual cosine scores for a candidate
set.  A recurrent evidence state is updated once per causal prefix; no
category, track ID, text, future or controller state is exposed.  The score is
an anchored residual on the raw cosine, so an untrained model is exactly the
raw policy.
"""
from __future__ import annotations

import torch
from torch import nn


class CausalMemoryScorer(nn.Module):
    """Stateful list scorer with a bounded raw-preserving residual."""

    delta_max = 0.08

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.hidden = int(hidden)
        # [raw, temporal delta, previous evidence, abs delta, rank, entropy, age]
        self.update = nn.Sequential(nn.Linear(7, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.evidence_head = nn.Linear(hidden, 1)
        self.gate_head = nn.Linear(hidden, 1)
        self.residual_head = nn.Sequential(nn.Linear(hidden + 4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # Exact raw fallback at construction; training must earn an intervention.
        nn.init.zeros_(self.evidence_head.weight); nn.init.zeros_(self.evidence_head.bias)
        nn.init.zeros_(self.gate_head.weight); nn.init.constant_(self.gate_head.bias, -4.0)
        nn.init.zeros_(self.residual_head[-1].weight); nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def _rank(raw: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(torch.argsort(raw, dim=-1, descending=True), dim=-1)
        return 1.0 - order.to(raw.dtype) / max(int(raw.shape[-1] - 1), 1)

    @staticmethod
    def _entropy(raw: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(raw * 10.0, dim=-1)
        return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)

    def forward(self, raw_sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        if raw_sequence.ndim != 2 or raw_sequence.shape[0] != 5 or raw_sequence.shape[1] == 0:
            raise ValueError(f"expected raw_sequence [5,N], got {tuple(raw_sequence.shape)}")
        raw = raw_sequence.float()
        evidence = torch.zeros_like(raw[0])
        previous = raw[0]
        outputs: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        for t in range(raw.shape[0]):
            current = raw[t]
            diff = current - previous if t else torch.zeros_like(current)
            rank = self._rank(current)
            entropy = self._entropy(current).expand_as(current)
            age = current.new_full(current.shape, float(t) / max(raw.shape[0] - 1, 1))
            feat = torch.stack([current, diff, evidence, diff.abs(), rank, entropy, age], dim=-1)
            token = self.update(feat)
            update = torch.tanh(self.evidence_head(token).squeeze(-1))
            gate = torch.sigmoid(self.gate_head(token).squeeze(-1))
            evidence = 0.85 * evidence + gate * (current + 0.1 * update)
            summary = torch.stack([
                evidence,
                evidence - evidence.mean(),
                current - current.mean(),
                current.new_full(current.shape, float(t + 1) / raw.shape[0]),
            ], dim=-1)
            residual = self.delta_max * torch.tanh(self.residual_head(torch.cat([token, summary], dim=-1)).squeeze(-1))
            # A small learned confidence controls intervention; residual_head is
            # zero-initialized and raw remains the exact step-0 output.
            final = current + gate * residual
            outputs.append(final); states.append(evidence); gates.append(gate); residuals.append(residual)
            previous = current
        return {
            "scores": torch.stack(outputs, dim=0),
            "raw": raw,
            "evidence": torch.stack(states, dim=0),
            "gate": torch.stack(gates, dim=0),
            "residual": torch.stack(residuals, dim=0),
        }

