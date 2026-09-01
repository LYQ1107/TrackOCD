"""RC-MS-OCD risk-calibrated multi-state controller."""
from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class RCMSOCD(nn.Module):
    """Frozen raw geometry plus a small candidate/action policy.

    Known prototypes are buffers, never optimizer parameters.  The policy sees
    only role-conditioned masks and causal candidate statistics.
    """

    def __init__(self, known_prototypes: torch.Tensor, active_mask: torch.Tensor,
                 hidden: int = 96, max_states: int = 16, known_bias: torch.Tensor | None = None):
        super().__init__()
        proto = F.normalize(known_prototypes.clone().float(), dim=-1)
        self.register_buffer("known_prototypes", proto)
        self.register_buffer("prototype_active_mask", active_mask.clone().bool())
        self.register_buffer("known_bias", torch.zeros(proto.shape[0]) if known_bias is None else known_bias.clone().float())
        self.raw_dim = int(proto.shape[-1]); self.known_count = int(proto.shape[0]); self.max_states = int(max_states)
        # Candidate features: raw/z cosine, quality, count, dispersion, age,
        # anchor-present, margin, same-track and same-video boolean guards.
        self.candidate_scorer = nn.Sequential(nn.Linear(11, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.birth_head = nn.Sequential(nn.Linear(self.raw_dim + 15 + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.defer_head = nn.Sequential(nn.Linear(15 + 1, 48), nn.GELU(), nn.Linear(48, 1))
        self.known_scale = 12.0
        self.birth_bias = nn.Parameter(torch.tensor(-0.25))
        self.defer_bias = nn.Parameter(torch.tensor(-0.50))
        # Fold-local calibration starts conservatively but remains reachable
        # on the frozen DINOv2 geometry; final thresholds are re-estimated from
        # supported-known validation streams only.
        self.tau_known = 0.20
        self.tau_assign = 0.52
        self.tau_ready = 0.45

    def prototype_hash(self) -> str:
        return hashlib.sha256(self.known_prototypes.detach().cpu().numpy().tobytes()).hexdigest()

    def embed(self, raw: torch.Tensor, geom: torch.Tensor, quality: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z = F.normalize(raw.float(), dim=-1)
        if quality is None:
            quality = torch.sigmoid(geom[..., 0])
        return {"z_raw": z, "z": z, "quality": quality.float()}

    def action_logits(self, emb: dict[str, torch.Tensor], known_mask: torch.Tensor,
                      state_bundle: dict[str, torch.Tensor], allow_defer: bool = True) -> dict[str, torch.Tensor]:
        raw = emb["z_raw"]; z = emb["z"]; q = emb["quality"]
        known_mask = known_mask.bool() & self.prototype_active_mask[None, :]
        known = self.known_scale * (z @ self.known_prototypes.T) + self.known_bias[None, :]
        known = known.masked_fill(~known_mask, -1e4)
        sr = state_bundle["state_raw"]; sz = state_bundle["state_z"]
        sf = state_bundle["state_features"]; sm = state_bundle["state_mask"]
        if sr.shape[1]:
            # StateMemory stores normalized raw/z prototypes after every
            # causal update; avoiding a second normalization removes a costly
            # per-item reduction without changing the represented vectors.
            raw_sim = torch.einsum("bd,bkd->bk", raw, sr)
            z_sim = torch.einsum("bd,bkd->bk", z, sz)
            best = raw_sim.masked_fill(~sm, -1e4).max(dim=1, keepdim=True).values
            top2 = raw_sim.masked_fill(~sm, -1e4).topk(min(2, raw_sim.shape[1]), dim=1).values
            second = top2[:, 1:2] if top2.shape[1] > 1 else torch.full_like(best, -1.0)
            margin = raw_sim - second
            admissible = sm.float()
            qcol = q[:, None].expand_as(raw_sim)
            feat = torch.cat([raw_sim[..., None], z_sim[..., None], qcol[..., None],
                              sf, margin[..., None], admissible[..., None]], dim=-1)
            # sf has 6 columns, giving 11 total columns.
            existing = self.candidate_scorer(feat).squeeze(-1)
            existing = existing.masked_fill(~sm, -1e4)
        else:
            existing = raw.new_zeros((raw.shape[0], 0))
        if existing.shape[1] < self.max_states:
            pad = raw.new_full((raw.shape[0], self.max_states - existing.shape[1]), -1e4)
            existing_padded = torch.cat([existing, pad], dim=1)
        else:
            existing_padded = existing[:, :self.max_states]
        birth_in = torch.cat([raw, torch.zeros(raw.shape[0], 15, device=raw.device, dtype=raw.dtype), q[:, None]], dim=-1)
        # Geometry is supplied separately by forward and copied into the input
        # through the temporary field to keep the scorer interface explicit.
        if "geom" in emb:
            birth_in = torch.cat([raw, emb["geom"], q[:, None]], dim=-1)
        new = self.birth_head(birth_in).squeeze(-1) + self.birth_bias
        defer = self.defer_head(torch.cat([emb.get("geom", torch.zeros(raw.shape[0], 15, device=raw.device)), q[:, None]], dim=-1)).squeeze(-1) + self.defer_bias
        if not allow_defer:
            defer = defer - 1e4
        logits = torch.cat([known, existing_padded, new[:, None], defer[:, None]], dim=-1)
        return {"logits": logits, "known_logits": known, "existing_logits": existing_padded,
                "new_logit": new, "defer_logit": defer, "candidate_raw_similarity":
                (raw_sim if sr.shape[1] else raw.new_zeros((raw.shape[0], 0))),
                "candidate_score": (existing if sr.shape[1] else raw.new_zeros((raw.shape[0], 0)))}

    def forward(self, raw: torch.Tensor, geom: torch.Tensor, quality: torch.Tensor,
                known_mask: torch.Tensor, state_bundle: dict[str, torch.Tensor],
                allow_defer: bool = True) -> dict[str, torch.Tensor]:
        emb = self.embed(raw, geom, quality); emb["geom"] = geom
        emb.update(self.action_logits(emb, known_mask, state_bundle, allow_defer=allow_defer))
        return emb


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {"total": int(sum(p.numel() for p in model.parameters())),
            "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}
