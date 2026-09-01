"""TrackOCD Phase51 unified causal MOT+OCD model.

The implementation is intentionally small enough to audit on four GPUs.  It
operates on the repository's key-aligned visual features and normalized causal
geometry; no category, text, semantic-ID or physical-ID feature enters the
forward path.  Physical IDs are only evaluator-side bookkeeping.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class UnifiedTrackOCD(nn.Module):
    """Joint proposal, association, track, correspondence and controller graph."""

    def __init__(self, raw_dim: int = 768, geom_dim: int = 15,
                 hidden: int = 192, support_hidden: int = 192) -> None:
        super().__init__()
        self.raw_dim = int(raw_dim)
        self.geom_dim = int(geom_dim)
        self.hidden = int(hidden)
        self.support_hidden = int(support_hidden)
        row_dim = self.raw_dim + self.geom_dim

        # Class-agnostic proposal/objectness and box refinement.
        self.proposal_encoder = nn.Sequential(
            nn.LayerNorm(row_dim), nn.Linear(row_dim, 128), nn.GELU(),
            nn.Linear(128, hidden), nn.GELU(),
        )
        self.objectness_head = nn.Linear(hidden, 1)
        self.proposal_quality_head = nn.Linear(hidden, 1)
        self.bbox_delta_head = nn.Linear(hidden, 4)

        # Causal track query and lifecycle heads.
        self.track_gru = nn.GRU(row_dim, hidden, num_layers=1, batch_first=True)
        self.track_query = nn.Linear(hidden, hidden)
        self.lifecycle_head = nn.Linear(hidden, 3)  # birth/continue/terminate

        # Raw-preserving semantic representation.  The anchor is the latest
        # causal visual vector; residual magnitude is explicitly bounded.
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden + self.raw_dim),
            nn.Linear(hidden + self.raw_dim, hidden), nn.GELU(),
            nn.Linear(hidden, self.raw_dim),
        )

        # Support-conditioned context and uncertainty/evidence heads.
        self.support_proj = nn.Sequential(
            nn.LayerNorm(self.raw_dim), nn.Linear(self.raw_dim, support_hidden), nn.GELU()
        )
        self.support_delta = nn.Sequential(
            nn.LayerNorm(self.raw_dim + support_hidden),
            nn.Linear(self.raw_dim + support_hidden, support_hidden), nn.GELU(),
            nn.Linear(support_hidden, self.raw_dim),
        )
        self.support_quality_head = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1)
        )

        # Physical association is differentiable and class agnostic.
        self.association_head = nn.Sequential(
            nn.LayerNorm(self.raw_dim * 2 + self.geom_dim * 2 + 4),
            nn.Linear(self.raw_dim * 2 + self.geom_dim * 2 + 4, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

        # Semantic state/controller share the same representation graph.
        controller_dim = self.raw_dim + support_hidden + 5
        self.controller = nn.Sequential(
            nn.LayerNorm(controller_dim), nn.Linear(controller_dim, 96), nn.GELU(),
            nn.Linear(96, 3),  # COMMIT / DEFER / RESET_REJECT
        )

        # Start close to the raw anchor while retaining gradients for the
        # support and semantic branches.
        nn.init.normal_(self.residual_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.support_delta[-1].weight)
        nn.init.zeros_(self.support_delta[-1].bias)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), dim=-1, eps=1e-6)

    def proposal(self, raw: torch.Tensor, geom: torch.Tensor) -> dict[str, torch.Tensor]:
        """Class-agnostic proposal outputs for a causal row/sequence."""
        h = self.proposal_encoder(torch.cat([raw.float(), geom.float()], dim=-1))
        return {
            "objectness_logit": self.objectness_head(h).squeeze(-1),
            "proposal_quality_logit": self.proposal_quality_head(h).squeeze(-1),
            "bbox_delta": 0.25 * torch.tanh(self.bbox_delta_head(h)),
            "proposal_hidden": h,
        }

    def association(self, raw_a: torch.Tensor, geom_a: torch.Tensor,
                    raw_b: torch.Tensor, geom_b: torch.Tensor) -> torch.Tensor:
        """Pairwise same-physical-track logit without identity features."""
        ra = self._normalize(raw_a)
        rb = self._normalize(raw_b)
        delta = (rb - ra).abs()
        geom_delta = (geom_b.float() - geom_a.float()).abs()
        pair = torch.cat([ra, rb, geom_a.float(), geom_b.float(),
                          delta[..., :4], geom_delta[..., :0]], dim=-1)
        # The declared input dimension includes four compact geometry terms;
        # use the first four normalized coordinates explicitly.
        pair = torch.cat([ra, rb, geom_a.float(), geom_b.float(), delta[..., :4]], dim=-1)
        return self.association_head(pair).squeeze(-1)

    def encode_sequence(self, raw_seq: torch.Tensor, geom_seq: torch.Tensor,
                        mask: Optional[torch.Tensor] = None,
                        support: Optional[torch.Tensor] = None,
                        support_mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """Encode a causal track prefix and optional strictly-prior support set."""
        raw_seq = raw_seq.float()
        geom_seq = geom_seq.float()
        bsz, timesteps, _ = raw_seq.shape
        if mask is None:
            mask = torch.ones((bsz, timesteps), dtype=torch.bool, device=raw_seq.device)
        row_in = torch.cat([raw_seq, geom_seq], dim=-1)
        hseq, hlast = self.track_gru(row_in)
        lengths = mask.long().sum(1).clamp_min(1) - 1
        h = hseq[torch.arange(bsz, device=raw_seq.device), lengths]
        latest = raw_seq[torch.arange(bsz, device=raw_seq.device), lengths]
        anchor = self._normalize(latest)
        residual = 0.05 * torch.tanh(self.residual_head(torch.cat([h, anchor], dim=-1)))
        base = self._normalize(anchor + residual)

        if support is None:
            support = base.new_zeros((bsz, 0, self.raw_dim))
        support = support.float()
        if support_mask is None:
            support_mask = torch.ones(support.shape[:2], dtype=torch.bool, device=base.device)
        support_mask = support_mask.bool()
        if support.shape[1] == 0:
            has_support = torch.zeros((bsz,), dtype=torch.bool, device=base.device)
            support_ctx = base.new_zeros((bsz, self.support_hidden))
            support_raw = base.new_zeros((bsz, self.raw_dim))
            max_sim = base.new_zeros((bsz,))
            support_count = base.new_zeros((bsz,))
        else:
            support_norm = self._normalize(support)
            sim = torch.einsum("bd,bsd->bs", base, support_norm)
            sim = sim.masked_fill(~support_mask, -1e4)
            max_sim = sim.max(dim=1).values.clamp(-1., 1.)
            weights = torch.softmax(sim, dim=1) * support_mask.float()
            denom = weights.sum(1, keepdim=True).clamp_min(1e-6)
            weights = weights / denom
            support_raw = torch.einsum("bs,bsd->bd", weights, support_norm)
            support_ctx = self.support_proj(support_raw)
            support_count = support_mask.float().sum(1)
            has_support = support_count > 0

        support_fraction = (support_count / max(float(support.shape[1]), 1.0)).clamp(0., 1.)
        quality_inputs = torch.stack([
            max_sim, support_fraction, torch.sigmoid(h[..., 0]),
            torch.sigmoid(h[..., 1]),
        ], dim=-1)
        support_quality_logit = self.support_quality_head(quality_inputs).squeeze(-1)
        support_quality = torch.sigmoid(support_quality_logit)
        context_delta = 0.05 * torch.tanh(self.support_delta(torch.cat([base, support_ctx], dim=-1)))
        semantic_with_support = self._normalize(base + context_delta)
        # Exact fallback is part of the contract: no/invalid support returns
        # exactly the normalized raw anchor, independent of learned weights.
        semantic = torch.where(has_support[:, None], semantic_with_support, anchor)

        evidence = torch.where(has_support, (max_sim + 1.) * 0.5 * support_quality, base.new_zeros(()).expand_as(support_quality))
        persistence = torch.where(has_support, support_fraction, base.new_zeros(()).expand_as(support_fraction))
        uncertainty = 1. - support_quality
        contradiction = torch.where(has_support, (1. - max_sim) * 0.5, base.new_ones(()).expand_as(max_sim))
        state_features = torch.stack([evidence, persistence, uncertainty, contradiction, support_quality], dim=-1)
        controller_input = torch.cat([semantic, support_ctx, state_features], dim=-1)
        action_logits = self.controller(controller_input)
        proposal = self.proposal(raw_seq.reshape(-1, self.raw_dim), geom_seq.reshape(-1, self.geom_dim))
        proposal = {k: v.reshape(bsz, timesteps, *v.shape[1:]) for k, v in proposal.items()}
        lifecycle_logits = self.lifecycle_head(hseq)
        return {
            "anchor": anchor,
            "raw": anchor,
            "track_query": self.track_query(h),
            "residual": residual,
            "semantic": semantic,
            "support_context": support_ctx,
            "support_quality": support_quality,
            "support_quality_logit": support_quality_logit,
            "support_similarity": max_sim,
            "support_count": support_count,
            "state_features": state_features,
            "action_logits": action_logits,
            "lifecycle_logits": lifecycle_logits,
            **proposal,
        }


def metadata(model: UnifiedTrackOCD) -> dict[str, object]:
    return {
        "architecture": "class-agnostic proposal + differentiable association + causal GRU query + support memory + semantic state/controller",
        "raw_dim": model.raw_dim, "geom_dim": model.geom_dim,
        "hidden": model.hidden, "support_hidden": model.support_hidden,
        "semantic_dim": model.raw_dim, "actions": ["COMMIT", "DEFER", "RESET_REJECT"],
        "forbidden_inputs": ["category_name", "category_text", "semantic_id", "physical_id", "future_frame", "future_track", "held_gt", "StateMemory", "controller_action"],
        "raw_fallback": "exact normalized latest causal visual anchor when support mask is empty",
    }
