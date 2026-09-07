"""Causal persistent runtime shared by held replay and TRAIN rollout."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch

from src.iclr27_phase88.controller import CausalPersistentOCD, decode_joint_action, mask_joint_logits
from src.iclr27_phase88.data import FeatureStore
from src.iclr27_phase88.memory import StateMemoryV2, TargetSession
from src.iclr27_phase88.support_evidence import compute_support_evidence
from src.iclr27_phase88.transitions import apply_session_action, finalize_session


class CausalPersistentRuntime:
    def __init__(self, model: CausalPersistentOCD, store: FeatureStore,
                 device: torch.device | str = "cpu") -> None:
        self.model = model
        self.store = store
        self.device = torch.device(device)
        self.memory = StateMemoryV2(max_states=model.max_states, max_prototypes=4,
                                    raw_dim=768, device=self.device)

    def reset_stream(self) -> None:
        self.memory.reset()

    def _process(self, track_key: str, video_id: int, known_mask: torch.Tensor,
                 support_features: torch.Tensor | None = None,
                 oracle_category_for_eval: int | None = None,
                 forced_actions: Iterable[tuple[str, int | None, int | None]] | None = None,
                 retain_grad: bool = False,
                 support_mode: bool = True) -> dict[str, Any]:
        array = self.store.track(track_key)
        n = len(array.raw)
        if n == 0:
            return {"track_key": track_key, "video_id": int(video_id), "trace": [],
                    "first_commit": None, "final_action": None, "reset_count": 0,
                    "global_state_count": len(self.memory.states), "source_tracks_processed": 0}
        raw = torch.from_numpy(array.raw[None]).to(self.device)
        geom = torch.from_numpy(array.geom[None]).to(self.device)
        quality = torch.from_numpy(array.quality[None]).to(self.device)
        mask = torch.ones((1, n), dtype=torch.bool, device=self.device)
        encoded = self.model.encode_track(raw, geom, mask)
        km = known_mask.to(self.device).bool().reshape(1, -1)
        session = TargetSession(track_key, int(video_id))
        trace: list[dict[str, Any]] = []
        forced = list(forced_actions) if forced_actions is not None else None
        context = torch.enable_grad() if retain_grad else torch.no_grad()
        with context:
            for position in range(n):
                tensors = self.memory.build_candidate_tensors(video_id, track_key)
                previous = session.evidence if session.evidence is not None else torch.zeros((1, self.model.max_states), device=self.device)
                q = quality[:, position]
                if not support_mode:
                    support = torch.zeros((1, 8), device=self.device)
                elif support_features is not None:
                    support = support_features.to(self.device).reshape(1, 8).expand(1, -1)
                else:
                    support = compute_support_evidence(
                        encoded["semantic_seq"][:, position], tensors["prototypes"],
                        tensors["prototype_mask"], tensors["state_stats"], q,
                        torch.tensor([session.best_streak], device=self.device),
                    )
                output = self.model.forward_action(
                    encoded["semantic_seq"][:, position], encoded["track_hidden_seq"][:, position],
                    tensors["prototypes"], tensors["prototype_mask"], tensors["state_stats"],
                    previous, support, q, torch.tensor([session.best_streak], device=self.device), km,
                )
                logits = mask_joint_logits(
                    output["joint_logits"], self.model.known_count, self.model.max_states,
                    [session.committed_action], [session.committed_state_index],
                    [session.committed_known_index],
                )
                action, slot, confidence = decode_joint_action(logits, self.model.known_count, self.model.max_states)
                known_index = slot if action == "KNOWN" else None
                if forced is not None and position < len(forced):
                    forced_action, forced_slot, forced_known = forced[position]
                    action = str(forced_action)
                    slot = forced_slot if action == "EXISTING" else None
                    known_index = forced_known if action == "KNOWN" else None
                    confidence = 1.0
                global_index = None
                sid = None
                if action == "EXISTING" and slot is not None and slot < len(tensors["state_indices"]):
                    global_index = int(tensors["state_indices"][slot])
                    sid = int(self.memory.states[global_index].sid)
                valid = tensors["prototype_mask"].any(dim=-1).squeeze(0)
                best = int(torch.argmax(output["state_logits"], dim=-1).item()) if bool(valid.any()) else None
                if best is None:
                    session.best_streak = 0
                    session.previous_best_index = None
                elif best == session.previous_best_index:
                    session.best_streak += 1
                else:
                    session.best_streak = 1
                    session.previous_best_index = best
                apply_session_action(session, action, slot, known_index, global_index)
                if session.evidence is None:
                    session.evidence = torch.zeros_like(previous)
                session.evidence = 0.70 * session.evidence + 0.30 * torch.sigmoid(output["state_logits"])
                session.collected_vectors.append(encoded["semantic_seq"][0, position].detach().cpu().numpy())
                trace.append({
                    "position": position + 1,
                    "action": action,
                    "predicted": action,
                    "semantic_id": sid,
                    "state_index": slot if action == "EXISTING" else None,
                    "global_state_index": global_index,
                    "known_index": known_index,
                    "confidence": confidence,
                    "quality": float(q.detach().cpu().item()),
                    "joint_logits": logits.squeeze(0).detach().cpu().tolist(),
                    "state_logits": output["state_logits"].squeeze(0).detach().cpu().tolist(),
                    "evidence": session.evidence.squeeze(0).detach().cpu().tolist(),
                    "best_streak": int(session.best_streak),
                    "global_state_count": len(self.memory.states),
                    "session_committed_action": session.committed_action,
                    "candidate_sids": tensors["candidate_sids"],
                })
        if session.collected_vectors:
            final_vector = torch.from_numpy(np.mean(np.asarray(session.collected_vectors, np.float32), axis=0)).to(self.device)
            finalize_session(self.memory, session, final_vector, len(session.collected_vectors), oracle_category_for_eval)
        first = next((row for row in trace if row["action"] in {"KNOWN", "EXISTING", "NEW"}), None)
        return {
            "track_key": track_key,
            "video_id": int(video_id),
            "trace": trace,
            "first_commit": first,
            "final_action": session.committed_action,
            "reset_count": int(session.reset_count),
            "global_state_count": len(self.memory.states),
            "source_tracks_processed": 1,
            "session": session,
        }

    def process_track(self, track_key: str, video_id: int, known_mask: torch.Tensor,
                      support_features: torch.Tensor | None = None,
                      oracle_category_for_eval: int | None = None,
                      forced_actions: Iterable[tuple[str, int | None, int | None]] | None = None,
                      retain_grad: bool = False,
                      support_mode: bool = True) -> dict[str, Any]:
        return self._process(track_key, video_id, known_mask, support_features,
                             oracle_category_for_eval, forced_actions, retain_grad, support_mode)
