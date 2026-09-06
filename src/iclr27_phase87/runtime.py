"""Online monotonic runtime with transactional track-end state updates."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.iclr27_phase87.controller import CausalPersistentOCD, decode_joint_action, mask_joint_logits
from src.iclr27_phase87.data import FeatureStore
from src.iclr27_phase87.memory import StateMemoryV2, TargetSession


class CausalPersistentRuntime:
    def __init__(self, model: CausalPersistentOCD, store: FeatureStore, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.store = store
        self.device = torch.device(device)
        self.memory = StateMemoryV2(max_states=model.max_states, device=self.device)

    def reset_stream(self) -> None:
        self.memory.reset()

    @torch.no_grad()
    def seed_track(self, track_key: str, video_id: int) -> None:
        array = self.store.track(track_key)
        raw = torch.from_numpy(array.raw[None]).to(self.device)
        geom = torch.from_numpy(array.geom[None]).to(self.device)
        mask = torch.ones((1, len(array.raw)), dtype=torch.bool, device=self.device)
        encoded = self.model.encode_track(raw, geom, mask)
        self.memory._new_state(encoded["semantic"].squeeze(0), video_id, track_key)

    @torch.no_grad()
    def process_track(self, track_key: str, video_id: int, support_features: torch.Tensor | None = None) -> dict[str, Any]:
        array = self.store.track(track_key)
        length = len(array.raw)
        raw = torch.from_numpy(array.raw[None]).to(self.device)
        geom = torch.from_numpy(array.geom[None]).to(self.device)
        quality = torch.from_numpy(array.quality[None]).to(self.device)
        mask = torch.ones((1, length), dtype=torch.bool, device=self.device)
        encoded = self.model.encode_track(raw, geom, mask)
        support = support_features.to(self.device).reshape(1, 8) if support_features is not None else torch.zeros((1, 8), device=self.device)
        session = TargetSession(track_key, int(video_id))
        trace = []
        for position in range(length):
            tensors = self.memory.build_candidate_tensors(video_id, track_key)
            previous = session.evidence if session.evidence is not None else torch.zeros((1, self.model.max_states), device=self.device)
            quality_t = quality[:, position]
            output = self.model.forward_action(encoded["semantic_seq"][:, position], encoded["track_hidden_seq"][:, position], tensors["prototypes"], tensors["prototype_mask"], tensors["state_stats"], previous, support, quality_t, torch.tensor([session.best_streak], device=self.device))
            logits = mask_joint_logits(output["joint_logits"], self.model.known_count, self.model.max_states, [session.committed_action], [session.committed_state_index])
            action, state_slot, confidence = decode_joint_action(logits, self.model.known_count, self.model.max_states)
            state_scores = output["state_logits"].squeeze(0)
            session.evidence = 0.70 * previous + 0.30 * torch.sigmoid(output["state_logits"])
            best = int(torch.argmax(state_scores).item()) if len(tensors["state_indices"]) else None
            session.best_streak = session.best_streak + 1 if best is not None and best == session.previous_best_index else 1
            session.previous_best_index = best
            if session.committed_action is None and action in {"EXISTING", "NEW", "KNOWN"}:
                session.committed_action = "EXISTING" if action in {"EXISTING", "KNOWN"} else "NEW"
                session.committed_state_index = state_slot
            elif action == "RESET":
                session.reset()
            session.collected_vectors.append(encoded["semantic_seq"][0, position].detach().cpu().numpy())
            trace.append({"position": position + 1, "action": action, "semantic_id": None, "state_index": state_slot, "known_index": state_slot if action == "KNOWN" else None, "confidence": confidence, "joint_logits": logits.squeeze(0).cpu().tolist(), "state_logits": output["state_logits"].squeeze(0).cpu().tolist(), "evidence": session.evidence.squeeze(0).cpu().tolist(), "best_streak": session.best_streak, "global_state_count": len(self.memory.states), "session_committed_action": session.committed_action})
        if session.collected_vectors:
            final_vector = torch.from_numpy(np.mean(np.asarray(session.collected_vectors, np.float32), axis=0)).to(self.device)
            self.memory.finalize_track(session, final_vector, len(session.collected_vectors))
        return {"track_key": track_key, "video_id": int(video_id), "trace": trace, "first_commit": next((row for row in trace if row["action"] in {"KNOWN", "EXISTING", "NEW"}), None), "final_action": session.committed_action, "reset_count": session.reset_count, "global_state_count": len(self.memory.states)}
