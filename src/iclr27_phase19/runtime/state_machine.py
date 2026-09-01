"""Single causal memory/controller used by training and inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def decode_action_index(index: int, known_count: int, state_count: int,
                       max_states: int) -> tuple[str, int | None]:
    """Decode the padded controller action space used by every caller."""
    if index < known_count:
        return "KNOWN", index
    if index < known_count + state_count:
        return "EXISTING", index - known_count
    if index == known_count + max_states:
        return "NEW", None
    return "DEFER", None


def blend_state(raw_old: torch.Tensor, z_old: torch.Tensor,
                raw_new: torch.Tensor, z_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The registered causal prototype update, shared by train/inference."""
    raw = torch.nn.functional.normalize(.8 * raw_old + .2 * raw_new.detach(), dim=-1)
    z = torch.nn.functional.normalize(.8 * z_old + .2 * z_new.detach(), dim=-1)
    return raw, z


@dataclass
class SemanticState:
    sid: int
    raw: torch.Tensor
    z: torch.Tensor
    video: int
    track_key: str
    count: int = 1
    age: int = 0
    anchors: list[np.ndarray] = field(default_factory=list)
    # Oracle-only training metadata; never passed to the model.
    oracle_category: int | None = None


class CausalStateMachine:
    """Action/memory semantics shared by rollout training and deployment."""

    def __init__(self, model: Any, known_count: int, max_states: int = 32,
                 allow_defer: bool = True, tau_ready: float = .45):
        self.model = model
        self.known_count = int(known_count)
        self.max_states = int(max_states)
        self.allow_defer = bool(allow_defer)
        self.tau_ready = float(tau_ready)
        self.states: list[SemanticState] = []
        self.next_sid = 100000
        self.step_index = 0
        self.trace: list[dict[str, Any]] = []

    def _state_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.states:
            z = torch.zeros(1, 0, self.model.raw_dim, device=device)
            raw = torch.zeros_like(z)
            mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
            return z, raw, mask
        z = torch.stack([s.z for s in self.states], dim=0)[None].to(device)
        raw = torch.stack([s.raw for s in self.states], dim=0)[None].to(device)
        mask = torch.ones(1, len(self.states), dtype=torch.bool, device=device)
        return z, raw, mask

    def predict(self, raw: torch.Tensor, geom: torch.Tensor, video: int,
                track_key: str, quality_override: float | None = None) -> dict[str, Any]:
        """Predict one action from current memory, then update exactly once."""
        device = raw.device
        emb = self.model.embed(raw[None], geom[None])
        readiness = float(emb["quality"].item()) if quality_override is None else float(quality_override)
        if self.allow_defer and readiness < self.tau_ready:
            action, state_index = "DEFER", None
            logits = None
        else:
            sz, sr, sm = self._state_tensors(device)
            logits = self.model.action_logits(emb, sz, sr, sm, allow_defer=self.allow_defer)
            idx = int(torch.argmax(logits[0]).item())
            probs = torch.softmax(logits[0], dim=-1)
            action, state_index = decode_action_index(idx, self.known_count, len(self.states), len(self.states))
            confidence = float(probs[idx].item())
        if logits is None:
            confidence = 1.0 - readiness
        sid = None
        if action == "KNOWN" and state_index is not None:
            # KNOWN uses the model's compact prototype index internally; the
            # evaluator wrapper maps it back to the declared known namespace.
            sid = int(state_index)
        if action == "NEW":
            if len(self.states) >= self.max_states:
                action = "DEFER"
            else:
                sid = self.next_sid; self.next_sid += 1
                self.states.append(SemanticState(sid=sid, raw=raw.detach(), z=emb["z"][0].detach(),
                                                 video=int(video), track_key=track_key))
        elif action == "EXISTING" and state_index is not None:
            state = self.states[state_index]
            sid = state.sid
            state.raw, state.z = blend_state(state.raw, state.z, raw, emb["z"][0])
            state.count += 1; state.age = 0
        for s in self.states:
            s.age += 1
        rec = {"step": self.step_index, "action": action, "semantic_id": sid,
               "track_key": track_key, "video": int(video), "readiness": readiness,
               "confidence": confidence, "state_count": len(self.states),
               "candidate_count": len(self.states)}
        self.step_index += 1; self.trace.append(rec)
        return {"action": action, "semantic_id": sid, "confidence": confidence,
                "readiness": readiness, "embedding": emb, "trace": rec}

    def add_teacher_state(self, raw: torch.Tensor, z: torch.Tensor, video: int,
                          track_key: str, oracle_category: int | None = None) -> int | None:
        if len(self.states) >= self.max_states:
            return None
        sid = self.next_sid; self.next_sid += 1
        self.states.append(SemanticState(sid=sid, raw=raw.detach(), z=z.detach(), video=int(video),
                                         track_key=track_key, oracle_category=oracle_category))
        return sid

    def replay_actions(self, actions: list[dict[str, Any]], tensors: list[tuple[torch.Tensor, torch.Tensor]]) -> list[dict[str, Any]]:
        out = []
        for a, (raw, geom) in zip(actions, tensors):
            got = self.predict(raw, geom, int(a["video"]), str(a["track_key"]))
            out.append({"expected_action": a["action"], "observed_action": got["action"],
                        "expected_state_count": a.get("state_count"), "observed_state_count": len(self.states)})
        return out
