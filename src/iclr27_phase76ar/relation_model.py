"""Raw-preserving, bank-aware selective local relation model."""
from __future__ import annotations

import torch
from torch import nn


class SelectiveAnchoredRelation(nn.Module):
    """Candidate-level residual with a query-bank abstention gate.

    The module never emits a replacement embedding.  A bounded ``tanh``
    residual is added to the supplied raw cosine only when both the bank and
    pair gates permit intervention.  The bank gate is initialized near zero,
    making raw scoring the exact step-0 policy up to a zero residual.
    """

    pair_dim = 1536
    summary_dim = 13
    quality_dim = 5
    bank_dim = 8
    delta_max = 0.10

    def __init__(self) -> None:
        super().__init__()
        self.pair_token = nn.Sequential(
            nn.Linear(self.pair_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.quality = nn.Sequential(nn.Linear(self.quality_dim, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
        self.pair_gate = nn.Sequential(nn.Linear(128 + self.summary_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.delta_head = nn.Sequential(nn.Linear(128 + self.summary_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.bank_gate = nn.Sequential(nn.Linear(self.bank_dim, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.delta_head[-1].weight); nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.constant_(self.bank_gate[-1].bias, -6.0)
        nn.init.zeros_(self.bank_gate[-1].weight)

    @staticmethod
    def bank_context(raw_scores: torch.Tensor) -> torch.Tensor:
        if raw_scores.ndim != 1 or raw_scores.numel() == 0:
            return raw_scores.new_zeros(8)
        ordered, _ = torch.sort(raw_scores, descending=True)
        top1 = ordered[0]
        top2 = ordered[1] if ordered.numel() > 1 else top1
        top3 = ordered[2] if ordered.numel() > 2 else top2
        centered = raw_scores - raw_scores.mean()
        probs = torch.softmax(raw_scores, dim=0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
        return torch.stack([
            top1, top2, top1 - top2, top1 - top3,
            raw_scores.mean(), raw_scores.std(unbiased=False), entropy,
            raw_scores.new_tensor(float(raw_scores.numel()) / 16.0),
        ])

    def forward_features(
        self,
        pair_tokens: torch.Tensor,
        quality_features: torch.Tensor,
        summary: torch.Tensor,
        raw_cosine: torch.Tensor | float,
        bank_raw_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if pair_tokens.ndim != 2 or pair_tokens.shape[-1] != self.pair_dim:
            raise ValueError(f"pair_tokens must be [N,1536], got {tuple(pair_tokens.shape)}")
        if quality_features.ndim != 2 or quality_features.shape[0] != pair_tokens.shape[0] or quality_features.shape[-1] != self.quality_dim:
            raise ValueError(f"quality_features must be [N,5], got {tuple(quality_features.shape)}")
        if summary.ndim != 1 or summary.shape[-1] != self.summary_dim:
            raise ValueError(f"summary must be [13], got {tuple(summary.shape)}")
        raw = torch.as_tensor(raw_cosine, dtype=pair_tokens.dtype, device=pair_tokens.device).reshape(())
        if pair_tokens.shape[0] == 0:
            z = raw.new_zeros(())
            return {"delta": z, "delta_bounded": z, "pair_gate": z, "bank_gate": z, "gate": z, "confidence": z, "final": raw, "weights": pair_tokens.new_zeros(0), "pooled": pair_tokens.new_zeros(128), "bank_context": pair_tokens.new_zeros(8), "bank_gate_logit": raw.new_zeros(())}
        token = self.pair_token(pair_tokens)
        weights = self.quality(quality_features).squeeze(-1)
        pooled = (weights.unsqueeze(-1) * token).sum(dim=0) / weights.sum().clamp_min(1e-6)
        context = torch.cat([pooled, summary], dim=-1)
        delta_raw = self.delta_head(context).squeeze(-1)
        delta_bounded = self.delta_max * torch.tanh(delta_raw)
        pair_gate_logit = self.pair_gate(context).squeeze(-1)
        pair_gate = torch.sigmoid(pair_gate_logit)
        bank_scores = bank_raw_scores if bank_raw_scores is not None else raw.reshape(1)
        bank_ctx = self.bank_context(bank_scores.to(pair_tokens))
        bank_gate_logit = self.bank_gate(bank_ctx).squeeze(-1)
        bank_gate = torch.sigmoid(bank_gate_logit)
        gate = bank_gate * pair_gate
        final = raw + gate * delta_bounded
        return {"delta": delta_raw, "delta_bounded": delta_bounded, "pair_gate": pair_gate, "bank_gate": bank_gate, "gate": gate, "confidence": gate, "final": final, "weights": weights, "pooled": pooled, "bank_context": bank_ctx, "bank_gate_logit": bank_gate_logit, "pair_gate_logit": pair_gate_logit}

    def forward(self, pair_tokens: torch.Tensor, quality_features: torch.Tensor, summary: torch.Tensor, raw_cosine: torch.Tensor | float, bank_raw_scores: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.forward_features(pair_tokens, quality_features, summary, raw_cosine, bank_raw_scores)
