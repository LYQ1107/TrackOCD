"""Persistent multi-state memory used identically by TRAIN and evaluation."""
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
    impurity_count: int = 0

    def snapshot(self) -> dict[str, Any]:
        valid = self.prototype_mask.detach().cpu().bool()
        return {
            "sid": int(self.sid),
            "birth_video": int(self.birth_video),
            "birth_track": str(self.birth_track),
            "total_tracks": int(self.total_tracks),
            "total_observations": int(self.total_observations),
            "age": int(self.age),
            "dispersion": float(self.dispersion),
            "oracle_birth_category": self.oracle_birth_category,
            "impurity_count": int(self.impurity_count),
            "prototype_count": int(valid.sum().item()),
            "prototype_counts": self.prototype_counts.detach().cpu().tolist(),
        }


@dataclass
class TargetSession:
    track_key: str
    video_id: int
    evidence: torch.Tensor | None = None
    previous_best_index: int | None = None
    best_streak: int = 0
    committed_action: str | None = None
    committed_state_index: int | None = None  # local candidate slot
    committed_global_index: int | None = None  # global memory index
    committed_known_index: int | None = None
    provisional_new: bool = False
    reset_count: int = 0
    missing_positive_state: bool = False
    reset_reason: str | None = None
    reset_injection_applied: bool = False
    reset_injection_global_index: int | None = None
    collected_vectors: list[np.ndarray] = field(default_factory=list)

    def reset(self, reason: str | None = None) -> None:
        self.evidence = None
        self.previous_best_index = None
        self.best_streak = 0
        self.committed_action = None
        self.committed_state_index = None
        self.committed_global_index = None
        self.committed_known_index = None
        self.provisional_new = False
        self.collected_vectors.clear()
        self.reset_count += 1
        self.reset_reason = reason


class StateMemoryV2:
    def __init__(self, max_states: int = 16, max_prototypes: int = 4,
                 raw_dim: int = 768, device: torch.device | str = "cpu") -> None:
        self.max_states = int(max_states)
        self.max_prototypes = int(max_prototypes)
        self.raw_dim = int(raw_dim)
        self.device = torch.device(device)
        self.states: list[SemanticStateV2] = []
        self.next_sid = 0

    def reset(self) -> None:
        self.states.clear()
        self.next_sid = 0

    def candidate_indices(self, video_id: int, track_key: str) -> list[int]:
        return [
            i for i, state in enumerate(self.states)
            if state.birth_video != int(video_id) and state.birth_track != str(track_key)
        ]

    def build_candidate_tensors(self, video_id: int, track_key: str) -> dict[str, Any]:
        indices = self.candidate_indices(video_id, track_key)[: self.max_states]
        prototypes = torch.zeros((1, self.max_states, self.max_prototypes, self.raw_dim), dtype=torch.float32, device=self.device)
        masks = torch.zeros((1, self.max_states, self.max_prototypes), dtype=torch.bool, device=self.device)
        stats = torch.zeros((1, self.max_states, 6), dtype=torch.float32, device=self.device)
        for slot, state_index in enumerate(indices):
            state = self.states[state_index]
            count = min(self.max_prototypes, int(state.prototypes.shape[0]))
            prototypes[0, slot, :count] = state.prototypes[:count].to(self.device)
            masks[0, slot, :count] = state.prototype_mask[:count].to(self.device)
            valid_counts = state.prototype_counts[:count]
            mean_count = float(valid_counts.mean()) if count else 0.0
            stats[0, slot] = torch.tensor([
                state.total_tracks / 32.0,
                state.total_observations / 64.0,
                state.age / 32.0,
                float(np.clip(state.dispersion, 0.0, 1.0)),
                count / float(self.max_prototypes),
                mean_count / 32.0,
            ], device=self.device)
        return {
            "prototypes": prototypes,
            "prototype_mask": masks,
            "state_stats": stats,
            "state_indices": indices,
            "candidate_sids": [int(self.states[i].sid) for i in indices],
        }

    def _new_state(self, vector: torch.Tensor, video_id: int, track_key: str,
                   observations: int, oracle_category: int | None = None) -> int | None:
        if len(self.states) >= self.max_states:
            return None
        vector = F.normalize(vector.detach().to(self.device).float(), dim=-1)
        prototypes = torch.zeros((self.max_prototypes, self.raw_dim), device=self.device)
        counts = torch.zeros((self.max_prototypes,), device=self.device)
        mask = torch.zeros((self.max_prototypes,), dtype=torch.bool, device=self.device)
        prototypes[0] = vector
        counts[0] = 1.0
        mask[0] = True
        self.states.append(SemanticStateV2(
            sid=self.next_sid,
            prototypes=prototypes,
            prototype_counts=counts,
            prototype_mask=mask,
            birth_video=int(video_id),
            birth_track=str(track_key),
            total_tracks=1,
            total_observations=int(observations),
            age=0,
            dispersion=0.0,
            oracle_birth_category=oracle_category,
        ))
        self.next_sid += 1
        return len(self.states) - 1

    def age_states(self, except_index: int | None = None) -> None:
        for i, state in enumerate(self.states):
            if except_index is not None and i == except_index:
                continue
            state.age += 1

    def update_existing(self, state_index: int, vector: torch.Tensor,
                        observations: int = 1, oracle_category: int | None = None) -> None:
        if state_index < 0 or state_index >= len(self.states):
            return
        state = self.states[state_index]
        vector = F.normalize(vector.detach().to(self.device).float(), dim=-1)
        valid = torch.where(state.prototype_mask)[0]
        if len(valid) < self.max_prototypes:
            slot = int(len(valid))
        else:
            sims = state.prototypes[valid] @ vector
            slot = int(valid[int(torch.argmax(sims))])
        old_count = float(state.prototype_counts[slot])
        beta = float(np.clip(1.0 / (state.total_tracks + 1.0), 0.05, 0.30))
        if old_count <= 0:
            state.prototypes[slot] = vector
            state.prototype_mask[slot] = True
            state.prototype_counts[slot] = 1.0
        else:
            best_sim = float((state.prototypes[state.prototype_mask] @ vector).max()) if bool(state.prototype_mask.any()) else 1.0
            novelty = float(1.0 - best_sim)
            state.dispersion = (1.0 - beta) * state.dispersion + beta * novelty
            state.prototypes[slot] = F.normalize((1.0 - beta) * state.prototypes[slot] + beta * vector, dim=-1)
            state.prototype_counts[slot] += 1.0
        state.total_tracks += 1
        state.total_observations += int(observations)
        state.age = 0
        if oracle_category is not None and state.oracle_birth_category is not None and int(oracle_category) != int(state.oracle_birth_category):
            state.impurity_count += 1

    def finalize_track(self, session: TargetSession, final_vector: torch.Tensor,
                       observations: int, oracle_category: int | None = None) -> None:
        """Commit global memory exactly once at physical track end."""
        action = session.committed_action
        if action == "EXISTING" and session.committed_global_index is not None:
            self.age_states(except_index=session.committed_global_index)
            self.update_existing(session.committed_global_index, final_vector, observations, oracle_category)
        elif action == "NEW":
            self.age_states()
            self._new_state(final_vector, session.video_id, session.track_key, observations, oracle_category)
        elif action == "KNOWN":
            self.age_states()
        else:
            self.age_states()

    def snapshot(self) -> list[dict[str, Any]]:
        return [s.snapshot() for s in self.states]
