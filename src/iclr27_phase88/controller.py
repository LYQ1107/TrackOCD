"""Phase88 causal joint action model with a real known branch."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.iclr27_phase88.model import CausalTrackEncoder, MemoryRelationEncoder


class CausalPersistentOCD(nn.Module):
    def __init__(self, known_prototypes: Tensor, active_known_mask: Tensor,
                 raw_dim: int = 768, geom_dim: int = 15, track_hidden: int = 256,
                 max_states: int = 16) -> None:
        super().__init__()
        if known_prototypes.ndim != 2 or known_prototypes.shape[1] != raw_dim:
            raise ValueError(f"known prototypes must be [N,{raw_dim}], got {tuple(known_prototypes.shape)}")
        if active_known_mask.numel() != known_prototypes.shape[0]:
            raise ValueError("active_known_mask/prototype count mismatch")
        self.max_states = int(max_states)
        self.known_count = int(known_prototypes.shape[0])
        self.track_encoder = CausalTrackEncoder(raw_dim, geom_dim, track_hidden)
        self.relation_encoder = MemoryRelationEncoder(128)
        self.register_buffer("known_prototypes", F.normalize(known_prototypes.float(), dim=-1))
        self.register_buffer("active_known_mask", active_known_mask.bool())
        context_dim = track_hidden + 15
        self.action_head = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, 128), nn.GELU(), nn.Linear(128, 3)
        )

    @property
    def action_dim(self) -> int:
        return self.known_count + self.max_states + 3

    def encode_track(self, raw_seq: Tensor, geom_seq: Tensor, mask: Tensor) -> dict[str, Tensor]:
        return self.track_encoder.forward_sequence(raw_seq, geom_seq, mask)

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
        sorted_logits = torch.sort(state_logits, dim=-1, descending=True).values
        best = sorted_logits[:, 0] if sorted_logits.shape[1] else semantic.new_full((semantic.shape[0],), -1e4)
        second = sorted_logits[:, 1] if sorted_logits.shape[1] > 1 else torch.full_like(best, -1e4)
        evidence = torch.sigmoid(state_logits)
        best_evidence = evidence.max(dim=-1).values if evidence.shape[1] else torch.zeros_like(best)
        streak = best_streak.float() / 16.0 if best_streak is not None else torch.zeros_like(best)
        memory_count = valid_states.float().sum(dim=-1) / float(self.max_states)
        context = torch.cat([
            track_hidden, quality.view(-1, 1), best.view(-1, 1), second.view(-1, 1),
            (best - second).view(-1, 1), best_evidence.view(-1, 1), streak.view(-1, 1),
            memory_count.view(-1, 1), support_features,
        ], dim=-1)
        action = self.action_head(context)
        logits = semantic.new_full((semantic.shape[0], self.action_dim), -1e4)
        active = self.active_known_mask[None, :]
        if known_mask is not None:
            active = active & known_mask.bool()
        known_logits = 12.0 * (F.normalize(semantic, dim=-1) @ self.known_prototypes.T)
        known_logits = known_logits.masked_fill(~active, -1e4)
        logits[:, : self.known_count] = known_logits
        logits[:, self.known_count:self.known_count + self.max_states] = state_logits
        logits[:, self.known_count + self.max_states:] = action
        return {
            "joint_logits": logits,
            "known_logits": known_logits,
            "state_logits": state_logits,
            "state_repr": relation["state_repr"],
            "context": context,
            "new_logit": action[:, 0],
            "defer_logit": action[:, 1],
            "reset_logit": action[:, 2],
        }


def mask_joint_logits(logits: Tensor, known_count: int, max_states: int,
                      committed_actions: Iterable[str | None],
                      committed_states: Iterable[int | None],
                      committed_known: Iterable[int | None] | None = None) -> Tensor:
    masked = logits.clone()
    # Iterables can be generators; materialize all once.
    actions = list(committed_actions)
    states = list(committed_states)
    knowns = list(committed_known) if committed_known is not None else [None] * len(actions)
    for row, (action, state, known) in enumerate(zip(actions, states, knowns)):
        new_index = known_count + max_states
        defer_index = new_index + 1
        reset_index = new_index + 2
        if action is None:
            masked[row, reset_index] = -1e4
        elif action == "EXISTING":
            allowed = {defer_index, reset_index}
            if state is not None and 0 <= int(state) < max_states:
                allowed.add(known_count + int(state))
            for index in range(masked.shape[1]):
                if index not in allowed:
                    masked[row, index] = -1e4
        elif action == "NEW":
            allowed = {new_index, defer_index, reset_index}
            for index in range(masked.shape[1]):
                if index not in allowed:
                    masked[row, index] = -1e4
        elif action == "KNOWN":
            allowed = {defer_index, reset_index}
            if known is not None and 0 <= int(known) < known_count:
                allowed.add(int(known))
            for index in range(masked.shape[1]):
                if index not in allowed:
                    masked[row, index] = -1e4
        else:
            raise ValueError(f"unknown committed action {action}")
    return masked


def decode_joint_action(logits: Tensor, known_count: int, max_states: int) -> tuple[str, int | None, float]:
    index = int(torch.argmax(logits, dim=-1).item())
    confidence = float(torch.softmax(logits, dim=-1).max().item())
    if index < known_count:
        return "KNOWN", index, confidence
    if index < known_count + max_states:
        return "EXISTING", index - known_count, confidence
    fixed = index - known_count - max_states
    return ("NEW", None, confidence) if fixed == 0 else (("DEFER", None, confidence) if fixed == 1 else ("RESET", None, confidence))
