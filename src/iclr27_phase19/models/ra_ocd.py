"""Identity-preserving rollout-aligned OCD model."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RAOCD(nn.Module):
    """Small adapter/controller retaining a direct raw DINOv2 path."""

    def __init__(self, known_prototypes: torch.Tensor, hidden: int = 256,
                 residual_rank: int = 64):
        super().__init__()
        self.raw_dim = int(known_prototypes.shape[-1])
        self.known_count = int(known_prototypes.shape[0])
        self.known_prototypes = nn.Parameter(F.normalize(known_prototypes.clone(), dim=-1))
        self.residual = nn.Sequential(
            nn.Linear(self.raw_dim + 15, hidden), nn.GELU(),
            nn.Linear(hidden, residual_rank), nn.GELU(),
            nn.Linear(residual_rank, self.raw_dim),
        )
        self.quality_head = nn.Sequential(nn.Linear(15, 64), nn.GELU(), nn.Linear(64, 1))
        self.known_bias = nn.Parameter(torch.zeros(self.known_count))
        self.action_scale = nn.Parameter(torch.tensor(8.0))
        self.new_bias = nn.Parameter(torch.tensor(0.0))
        self.defer_bias = nn.Parameter(torch.tensor(-0.5))
        # Zero gate gives an identity mapping at initialization.
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def embed(self, raw: torch.Tensor, geom: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = F.normalize(raw, dim=-1)
        delta = self.residual(torch.cat([raw, geom], dim=-1))
        gate = torch.tanh(self.residual_gate) * 0.25
        z_residual = gate * delta
        z = F.normalize(raw + z_residual, dim=-1)
        quality_logit = self.quality_head(geom).squeeze(-1)
        return {"z_raw": raw, "z_residual": z_residual, "z": z,
                "quality_logit": quality_logit, "quality": torch.sigmoid(quality_logit)}

    def action_logits(self, emb: dict[str, torch.Tensor], state_z: torch.Tensor,
                      state_raw: torch.Tensor, state_mask: torch.Tensor,
                      allow_defer: bool = True) -> torch.Tensor:
        """Scores [known classes, padded existing states, NEW, DEFER]."""
        z = emb["z"]
        raw = emb["z_raw"]
        scale = self.action_scale.clamp(1.0, 20.0)
        known = scale * (z @ F.normalize(self.known_prototypes, dim=-1).T) + self.known_bias
        if state_z.shape[1]:
            sem = torch.einsum("bd,bkd->bk", z, F.normalize(state_z, dim=-1))
            raw_sim = torch.einsum("bd,bkd->bk", raw, F.normalize(state_raw, dim=-1))
            existing = scale * (.25 * sem + .75 * raw_sim)
            existing = existing.masked_fill(~state_mask, -1e4)
        else:
            existing = z.new_zeros((z.shape[0], 0))
        new = (self.new_bias + .25 * emb["quality_logit"]).unsqueeze(-1)
        defer = (self.defer_bias - .50 * emb["quality_logit"]).unsqueeze(-1)
        if not allow_defer:
            defer = defer - 1e4
        return torch.cat([known, existing, new, defer], dim=-1)

    def forward(self, raw: torch.Tensor, geom: torch.Tensor, state_z: torch.Tensor,
                state_raw: torch.Tensor, state_mask: torch.Tensor,
                allow_defer: bool = True) -> dict[str, torch.Tensor]:
        emb = self.embed(raw, geom)
        emb["logits"] = self.action_logits(emb, state_z, state_raw, state_mask, allow_defer)
        emb["known_logits"] = emb["logits"][:, :self.known_count]
        return emb


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
