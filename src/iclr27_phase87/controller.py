"""Joint action decoder for KNOWN/EXISTING/NEW/DEFER/RESET."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.iclr27_phase87.model import CausalTrackEncoder, MemoryRelationEncoder


ACTION_NEW = "NEW"
ACTION_DEFER = "DEFER"
ACTION_RESET = "RESET"


class CausalPersistentOCD(nn.Module):
    def __init__(self, raw_dim: int = 768, geom_dim: int = 15, track_hidden: int = 256, max_states: int = 16, known_count: int = 0) -> None:
        super().__init__()
        self.max_states = max_states
        self.known_count = known_count
        self.track_encoder = CausalTrackEncoder(raw_dim, geom_dim, track_hidden)
        self.relation_encoder = MemoryRelationEncoder(128)
        self.register_buffer("known_prototypes", torch.zeros((known_count, raw_dim)))
        # 256 track features + 7 scalar state features + 8 support values = 271.
        # The prompt's 270-D arithmetic omitted one scalar; retaining all listed
        # causal inputs is the safer contract and is recorded in the report.
        context_dim = track_hidden + 15
        self.action_head = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, 128), nn.GELU(), nn.Linear(128, 3))

    @property
    def action_dim(self) -> int:
        return self.known_count + self.max_states + 3

    def encode_track(self, raw_seq: Tensor, geom_seq: Tensor, mask: Tensor) -> dict[str, Tensor]:
        return self.track_encoder.forward_sequence(raw_seq, geom_seq, mask)

    def forward_action(self, semantic: Tensor, track_hidden: Tensor, prototypes: Tensor, prototype_mask: Tensor, state_stats: Tensor, previous_evidence: Tensor, support_features: Tensor, quality: Tensor, best_streak: Tensor | None = None) -> dict[str, Tensor]:
        relation = self.relation_encoder(semantic, prototypes, prototype_mask, state_stats, previous_evidence, support_features, quality)
        state_logits = relation["state_logits"]
        # Invalid/empty prototype slots must not be eligible actions.  Keeping
        # this mask at the joint interface prevents a random relation-logit
        # from creating a state when causal memory has no candidate.
        valid_states = prototype_mask.any(dim=-1)
        state_logits = state_logits.masked_fill(~valid_states, -1e4)
        sorted_logits = torch.sort(state_logits.masked_fill(state_logits <= -1e3, -1e4), dim=-1, descending=True).values
        best = sorted_logits[:, 0] if sorted_logits.shape[1] else torch.full((semantic.shape[0],), -1e4, device=semantic.device)
        second = sorted_logits[:, 1] if sorted_logits.shape[1] > 1 else torch.full_like(best, -1e4)
        evidence = torch.sigmoid(state_logits)
        best_evidence = evidence.max(dim=-1).values if evidence.shape[1] else torch.zeros_like(best)
        streak = (best_streak.float() / 16.0) if best_streak is not None else torch.zeros_like(best)
        memory_count = prototype_mask.any(dim=-1).float().sum(dim=-1) / float(self.max_states)
        context = torch.cat([track_hidden, quality.view(-1, 1), best.view(-1, 1), second.view(-1, 1), (best - second).view(-1, 1), best_evidence.view(-1, 1), streak.view(-1, 1), memory_count.view(-1, 1), support_features], dim=-1)
        action = self.action_head(context)
        logits = torch.full((semantic.shape[0], self.action_dim), -1e4, dtype=semantic.dtype, device=semantic.device)
        if self.known_count:
            logits[:, : self.known_count] = 12.0 * (F.normalize(semantic, dim=-1) @ F.normalize(self.known_prototypes, dim=-1).T)
        logits[:, self.known_count : self.known_count + self.max_states] = state_logits
        logits[:, self.known_count + self.max_states :] = action
        return {"joint_logits": logits, "state_logits": state_logits, "state_repr": relation["state_repr"], "context": context, "new_logit": action[:, 0], "defer_logit": action[:, 1], "reset_logit": action[:, 2]}


def mask_joint_logits(logits: Tensor, known_count: int, max_states: int, committed_action: Iterable[str | None], committed_state: Iterable[int | None]) -> Tensor:
    masked = logits.clone()
    for row, (action, state) in enumerate(zip(committed_action, committed_state)):
        new_index, defer_index, reset_index = known_count + max_states, known_count + max_states + 1, known_count + max_states + 2
        if action is None:
            masked[row, reset_index] = -1e4
        elif action == "EXISTING":
            allowed = {known_count + int(state), defer_index, reset_index}
            for index in range(masked.shape[1]):
                if index not in allowed:
                    masked[row, index] = -1e4
        elif action == "NEW":
            allowed = {new_index, defer_index, reset_index}
            for index in range(masked.shape[1]):
                if index not in allowed:
                    masked[row, index] = -1e4
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
