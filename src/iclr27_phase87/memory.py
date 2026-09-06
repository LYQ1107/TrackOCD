"""Transactional multi-prototype semantic memory for Phase87."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SemanticStateV2:
    sid: int
    prototypes: torch.Tensor
    prototype_counts: torch.Tensor
    prototype_mask: torch.Tensor
    birth_video: int
    birth_track: str
    total_tracks: int = 1
    total_observations: int = 0
    age: int = 0
    dispersion: float = 0.0
    oracle_birth_category: int | None = None


@dataclass
class TargetSession:
    track_key: str
    video_id: int
    evidence: torch.Tensor | None = None
    previous_best_index: int | None = None
    best_streak: int = 0
    committed_action: str | None = None
    committed_state_index: int | None = None
    provisional_new: bool = False
    reset_count: int = 0
    collected_vectors: list[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self.evidence = None
        self.previous_best_index = None
        self.best_streak = 0
        self.committed_action = None
        self.committed_state_index = None
        self.provisional_new = False
        self.reset_count += 1
        self.collected_vectors.clear()


class StateMemoryV2:
    def __init__(self, max_states: int = 16, max_prototypes: int = 4, device: torch.device | str = "cpu") -> None:
        self.max_states = max_states
        self.max_prototypes = max_prototypes
        self.device = torch.device(device)
        self.states: list[SemanticStateV2] = []
        self.next_sid = 0

    def reset(self) -> None:
        self.states.clear()
        self.next_sid = 0

    def candidate_indices(self, video_id: int, track_key: str) -> list[int]:
        return [
            index for index, state in enumerate(self.states)
            if state.birth_video != int(video_id) and state.birth_track != str(track_key)
        ]

    def build_candidate_tensors(self, video_id: int, track_key: str) -> dict[str, Any]:
        indices = self.candidate_indices(video_id, track_key)
        prototypes = torch.zeros((1, self.max_states, self.max_prototypes, 768), dtype=torch.float32, device=self.device)
        masks = torch.zeros((1, self.max_states, self.max_prototypes), dtype=torch.bool, device=self.device)
        stats = torch.zeros((1, self.max_states, 6), dtype=torch.float32, device=self.device)
        for slot, state_index in enumerate(indices[: self.max_states]):
            state = self.states[state_index]
            count = min(self.max_prototypes, int(state.prototypes.shape[0]))
            prototypes[0, slot, :count] = state.prototypes[:count].to(self.device)
            masks[0, slot, :count] = state.prototype_mask[:count].to(self.device)
            mean_count = float(state.prototype_counts[:count].float().mean()) if count else 0.0
            stats[0, slot] = torch.tensor([state.total_tracks / 32.0, state.total_observations / 64.0, state.age / 32.0, np.clip(state.dispersion, 0.0, 1.0), count / float(self.max_prototypes), mean_count / 32.0], device=self.device)
        return {"prototypes": prototypes, "prototype_mask": masks, "state_stats": stats, "state_indices": indices[: self.max_states], "candidate_sids": [self.states[i].sid for i in indices[: self.max_states]]}

    def _new_state(self, vector: torch.Tensor, video_id: int, track_key: str) -> None:
        vector = F.normalize(vector.detach().to(self.device).float(), dim=-1)
        prototypes = torch.zeros((self.max_prototypes, 768), device=self.device)
        counts = torch.zeros((self.max_prototypes,), device=self.device)
        mask = torch.zeros((self.max_prototypes,), dtype=torch.bool, device=self.device)
        prototypes[0] = vector
        counts[0] = 1.0
        mask[0] = True
        self.states.append(SemanticStateV2(self.next_sid, prototypes, counts, mask, int(video_id), str(track_key)))
        self.next_sid += 1

    def update_existing(self, state_index: int, vector: torch.Tensor, observations: int = 1) -> None:
        if state_index < 0 or state_index >= len(self.states):
            return
        state = self.states[state_index]
        vector = F.normalize(vector.detach().to(self.device).float(), dim=-1)
        available = torch.where(state.prototype_mask)[0]
        if len(available) < self.max_prototypes:
            slot = int(len(available))
        else:
            sims = state.prototypes[available] @ vector
            slot = int(available[int(torch.argmax(sims))])
        old_count = float(state.prototype_counts[slot])
        alpha = float(np.clip(1.0 / (old_count + 1.0), 0.05, 0.30))
        if old_count <= 0:
            state.prototypes[slot] = vector
            state.prototype_mask[slot] = True
            state.prototype_counts[slot] = 1.0
        else:
            state.prototypes[slot] = F.normalize((1.0 - alpha) * state.prototypes[slot] + alpha * vector, dim=-1)
            state.prototype_counts[slot] += 1.0
        state.total_tracks += 1
        state.total_observations += int(observations)
        state.age += 1

    def finalize_track(self, session: TargetSession, final_track_prototype: torch.Tensor, observations: int) -> None:
        if session.committed_action == "EXISTING" and session.committed_state_index is not None:
            indices = self.candidate_indices(session.video_id, session.track_key)
            if session.committed_state_index < len(indices):
                self.update_existing(indices[session.committed_state_index], final_track_prototype, observations)
        elif session.committed_action == "NEW" and len(self.states) < self.max_states:
            self._new_state(final_track_prototype, session.video_id, session.track_key)
        # DEFER, RESET and unresolved tracks do not mutate global state.
