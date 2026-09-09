"""Hierarchical KNOWN/OPEN/DEFER/RESET controller built on Phase88 modules.

The route keeps the Phase88 physical stream, state memory and action masks.  It
only changes semantic arbitration: a router first chooses KNOWN, OPEN, DEFER
or RESET, and the OPEN branch then chooses an existing state or NEW.  The
assembled ``joint_logits`` retain the legacy action indices so the causal
runtime/evaluator can be reused without changing action semantics.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.iclr27_phase88.controller import CausalPersistentOCD


class OpenWorldRouter(nn.Module):
    def __init__(self, input_dim: int = 276, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class HierarchicalPersistentOCD(CausalPersistentOCD):
    """Phase88-compatible model with a trainable hierarchical router."""

    architecture = "h3_hierarchical_router"

    def __init__(self, known_prototypes: Tensor, active_known_mask: Tensor,
                 raw_dim: int = 768, geom_dim: int = 15, track_hidden: int = 256,
                 max_states: int = 16) -> None:
        super().__init__(known_prototypes, active_known_mask, raw_dim, geom_dim, track_hidden, max_states)
        # track hidden (256), quality (1), known summary (4), memory summary
        # (4), streak (1), memory age/dispersion (2), support (8).
        self.router_input_dim = track_hidden + 1 + 4 + 4 + 1 + 2 + 8
        self.open_world_router = OpenWorldRouter(self.router_input_dim, 128)

    @staticmethod
    def _summary(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        masked = values.masked_fill(~valid, -1e4)
        ordered = torch.sort(masked, dim=-1, descending=True).values
        best = ordered[:, 0] if ordered.shape[1] else values.new_zeros((values.shape[0],))
        second = ordered[:, 1] if ordered.shape[1] > 1 else torch.full_like(best, -1e4)
        margin = best - second
        count = valid.float().sum(dim=-1)
        return best, second, margin, count

    def forward_action(self, semantic: Tensor, track_hidden: Tensor, prototypes: Tensor,
                       prototype_mask: Tensor, state_stats: Tensor,
                       previous_evidence: Tensor, support_features: Tensor,
                       quality: Tensor, best_streak: Tensor | None = None,
                       known_mask: Tensor | None = None) -> dict[str, Tensor]:
        relation = self.relation_encoder(
            semantic, prototypes, prototype_mask, state_stats,
            previous_evidence, support_features, quality,
        )
        state_logits = relation["state_logits"]
        valid_states = prototype_mask.any(dim=-1)
        state_logits = state_logits.masked_fill(~valid_states, -1e4)
        state_best, state_second, state_margin, state_count = self._summary(state_logits, valid_states)
        state_evidence = torch.sigmoid(state_logits)
        state_best_evidence = state_evidence.masked_fill(~valid_states, 0.0).max(dim=-1).values if state_evidence.shape[1] else torch.zeros_like(state_best)
        streak = best_streak.float() / 16.0 if best_streak is not None else torch.zeros_like(state_best)
        valid_float = valid_states.float().unsqueeze(-1)
        age = (state_stats[:, :, 2:3] * valid_float).sum(dim=1) / state_count.clamp_min(1.0).unsqueeze(-1)
        dispersion = (state_stats[:, :, 3:4] * valid_float).sum(dim=1) / state_count.clamp_min(1.0).unsqueeze(-1)

        context = torch.cat([
            track_hidden, quality.view(-1, 1), state_best.view(-1, 1),
            state_second.view(-1, 1), state_margin.view(-1, 1),
            state_best_evidence.view(-1, 1), streak.view(-1, 1),
            (state_count / float(self.max_states)).view(-1, 1), support_features,
        ], dim=-1)
        action = self.action_head(context)

        active = self.active_known_mask[None, :]
        if known_mask is not None:
            active = active & known_mask.bool()
        known_logits = 12.0 * (F.normalize(semantic, dim=-1) @ self.known_prototypes.T)
        known_logits = known_logits.masked_fill(~active, -1e4)
        known_valid = active
        known_best, known_second, known_margin, known_count = self._summary(known_logits, known_valid)
        known_probs = torch.softmax(known_logits.masked_fill(~known_valid, -1e4), dim=-1)
        known_entropy = -(known_probs * torch.log(known_probs.clamp_min(1e-8))).sum(dim=-1)
        router_features = torch.cat([
            track_hidden, quality.view(-1, 1), known_best.view(-1, 1),
            known_second.view(-1, 1), known_margin.view(-1, 1), known_entropy.view(-1, 1),
            state_best.view(-1, 1), state_second.view(-1, 1), state_margin.view(-1, 1),
            (state_count / float(self.max_states)).view(-1, 1), streak.view(-1, 1),
            age, dispersion, support_features,
        ], dim=-1)
        router_logits = self.open_world_router(router_features)
        # OPEN's second stage uses the existing relation scores and the
        # baseline NEW score.  Router logits calibrate only the branch choice.
        open_logits = torch.cat([state_logits, action[:, 0:1]], dim=-1)
        logits = semantic.new_full((semantic.shape[0], self.action_dim), -1e4)
        logits[:, :self.known_count] = known_logits + router_logits[:, 0:1]
        logits[:, self.known_count:self.known_count + self.max_states] = state_logits + router_logits[:, 1:2]
        logits[:, self.known_count + self.max_states] = action[:, 0] + router_logits[:, 1]
        logits[:, self.known_count + self.max_states + 1] = router_logits[:, 2]
        logits[:, self.known_count + self.max_states + 2] = router_logits[:, 3]
        return {
            "joint_logits": logits,
            "known_logits": known_logits,
            "state_logits": state_logits,
            "state_repr": relation["state_repr"],
            "context": context,
            "router_logits": router_logits,
            "open_logits": open_logits,
            "known_entropy": known_entropy,
            "new_logit": action[:, 0],
            "defer_logit": action[:, 1],
            "reset_logit": action[:, 2],
        }
