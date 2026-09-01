"""Single causal state-transition core shared by train and inference.

The core intentionally keeps evaluator/trainer-only birth metadata on a state
object while exposing only raw/prototype/statistical tensors to the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class SemanticState:
    sid: int
    raw: torch.Tensor
    z: torch.Tensor
    birth_video: int
    birth_track: str
    count: int = 1
    dispersion: float = 0.0
    # Fast training mode keeps the causal dispersion update on-device.  The
    # scalar mirror is retained for evaluator snapshots and compatibility.
    dispersion_tensor: torch.Tensor | None = None
    age: int = 0
    anchors: list[np.ndarray] = field(default_factory=list)
    # Loss/evaluator-only metadata.  It is never a model feature.
    oracle_birth_category: int | None = None
    impurity_count: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "sid": int(self.sid), "count": int(self.count),
            "dispersion": float(self.dispersion_tensor.detach().cpu().item() if self.dispersion_tensor is not None else self.dispersion),
            "age": int(self.age), "birth_video": int(self.birth_video),
            "birth_track": str(self.birth_track),
            "oracle_birth_category": self.oracle_birth_category,
            "impurity_count": int(self.impurity_count),
            "raw": self.raw.detach().cpu().numpy().astype(np.float32).tolist(),
            "z": self.z.detach().cpu().numpy().astype(np.float32).tolist(),
            "anchor_count": len(self.anchors),
        }


def update_prototype(state: SemanticState, raw_new: torch.Tensor, z_new: torch.Tensor,
                     quality: float, confidence: float, *, max_anchors: int = 8,
                     allow_update: bool = True, fast_mode: bool = False,
                     store_anchor_values: bool = True) -> None:
    """Causal confidence/quality-gated EMA update.

    The EMA weight decreases with state count.  A rejected low-quality
    observation can influence a decision but cannot contaminate the committed
    prototype or anchor queue.
    """
    if not allow_update or quality < .35 or confidence < .55:
        state.age += 1
        return
    alpha = float(np.clip(1.0 / (state.count + 1.0), .05, .30))
    old_raw = state.raw
    old_z = state.z
    state.raw = F.normalize((1 - alpha) * old_raw + alpha * raw_new.detach(), dim=-1)
    state.z = F.normalize((1 - alpha) * old_z + alpha * z_new.detach(), dim=-1)
    sim_t = (old_raw.detach() * raw_new.detach()).sum()
    if fast_mode:
        prev = state.dispersion_tensor
        if prev is None:
            prev = torch.as_tensor(state.dispersion, dtype=sim_t.dtype, device=sim_t.device)
        state.dispersion_tensor = ((1 - alpha) * prev + alpha * (1.0 - sim_t)).detach()
    else:
        sim = float(sim_t.item())
        state.dispersion = float((1 - alpha) * state.dispersion + alpha * (1.0 - sim))
    state.count += 1
    state.age = 0
    if store_anchor_values:
        state.anchors.append(raw_new.detach().cpu().numpy().astype(np.float32))
    else:
        # Candidate logic uses only anchor presence/count during training; a
        # placeholder avoids a per-item GPU->CPU copy while preserving the
        # exact boolean feature and queue length semantics.
        state.anchors.append(np.empty((0,), dtype=np.float32))
    if len(state.anchors) > max_anchors:
        state.anchors = state.anchors[-max_anchors:]


class StateMemory:
    """Persistent anonymous memory with explicit reset boundaries."""

    def __init__(self, max_states: int = 16, max_anchors: int = 8, sid_start: int = 100000,
                 *, fast_mode: bool = False, record_trace: bool = True):
        self.max_states = int(max_states)
        self.max_anchors = int(max_anchors)
        self.sid_start = int(sid_start)
        self.fast_mode = bool(fast_mode)
        self.record_trace = bool(record_trace)
        self.reset()

    def reset(self) -> None:
        self.states: list[SemanticState] = []
        self.track_bindings: dict[str, int] = {}
        self.next_sid = int(self.sid_start)
        self.step_index = 0
        self.trace: list[dict[str, Any]] = []
        self._fast_raw_bank: torch.Tensor | None = None
        self._fast_z_bank: torch.Tensor | None = None
        self._fast_disp_bank: torch.Tensor | None = None

    @property
    def state_count(self) -> int:
        return len(self.states)

    def candidate_indices(self, video_id: int, track_key: str) -> list[int]:
        # Same physical track/video is not eligible to become a cross-track
        # semantic reuse candidate.  IDs are used only for this boolean rule.
        return [j for j, s in enumerate(self.states)
                if str(s.birth_track) == str(track_key)
                or (int(s.birth_video) != int(video_id) and str(s.birth_track) != str(track_key))]

    def build_candidate_tensors(self, raw: torch.Tensor, video_id: int, track_key: str,
                                device: torch.device | None = None) -> dict[str, Any]:
        """Build the exact candidate ordering consumed by model and evaluator."""
        device = device or raw.device
        indices = self.candidate_indices(video_id, track_key)
        d = int(raw.shape[-1])
        if not indices:
            return {"state_raw": torch.zeros(1, 0, d, device=device),
                    "state_z": torch.zeros(1, 0, d, device=device),
                    "state_features": torch.zeros(1, 0, 6, device=device),
                    "state_mask": torch.zeros(1, 0, dtype=torch.bool, device=device),
                    "state_indices": [], "candidate_sids": []}
        states = [self.states[i] for i in indices]
        if self.fast_mode and self._fast_raw_bank is not None and self._fast_z_bank is not None:
            state_raw = self._fast_raw_bank[indices][None]
            state_z = self._fast_z_bank[indices][None]
        else:
            state_raw = torch.stack([s.raw if s.raw.device == device else s.raw.to(device) for s in states], dim=0)[None]
            state_z = torch.stack([s.z if s.z.device == device else s.z.to(device) for s in states], dim=0)[None]
        if self.fast_mode:
            static = np.asarray([[min(s.count, 32) / 32., 0., min(s.age, 32) / 32.,
                                  1.0 if s.anchors else 0.0,
                                  1.0 if str(s.birth_track) == str(track_key) else 0.0,
                                  1.0 if int(s.birth_video) == int(video_id) else 0.0]
                                 for s in states], dtype=np.float32)
            feats = torch.from_numpy(static).to(device=device, dtype=state_raw.dtype)
            if self._fast_disp_bank is not None:
                disp = self._fast_disp_bank[indices]
            else:
                disp = torch.stack([(s.dispersion_tensor if s.dispersion_tensor is not None else
                                     torch.as_tensor(s.dispersion, dtype=state_raw.dtype, device=device))
                                    for s in states])
            feats[:, 1] = torch.clamp(disp, 0., 1.)
            feats = feats[None]
        else:
            feats = torch.tensor([[min(s.count, 32) / 32., min(s.dispersion, 1.), min(s.age, 32) / 32.,
                                   1.0 if s.anchors else 0.0,
                                   1.0 if str(s.birth_track) == str(track_key) else 0.0,
                                   1.0 if int(s.birth_video) == int(video_id) else 0.0]
                                  for s in states], dtype=torch.float32, device=device)[None]
        mask = torch.ones(1, len(states), dtype=torch.bool, device=device)
        return {"state_raw": state_raw, "state_z": state_z, "state_features": feats,
                "state_mask": mask, "state_indices": indices,
                "candidate_sids": [int(s.sid) for s in states]}

    def apply_action(self, action: str, raw: torch.Tensor, z: torch.Tensor,
                     video_id: int, track_key: str, *, state_index: int | None = None,
                     oracle_category: int | None = None, quality: float = 1.0,
                     confidence: float = 1.0, update_allowed: bool = True) -> dict[str, Any]:
        """Apply one discrete action and serialize the causal transition."""
        action = str(action); sid: int | None = None; candidate_index = state_index
        if action == "NEW":
            if len(self.states) >= self.max_states:
                action = "DEFER"
            else:
                sid = self.next_sid; self.next_sid += 1
                if self.fast_mode and self._fast_raw_bank is None:
                    self._fast_raw_bank = torch.empty((self.max_states, raw.shape[-1]), dtype=raw.dtype, device=raw.device)
                    self._fast_z_bank = torch.empty((self.max_states, z.shape[-1]), dtype=z.dtype, device=z.device)
                    self._fast_disp_bank = torch.zeros((self.max_states,), dtype=raw.dtype, device=raw.device)
                slot = len(self.states)
                if self.fast_mode:
                    self._fast_raw_bank[slot].copy_(raw.detach()); self._fast_z_bank[slot].copy_(z.detach())
                st = SemanticState(sid=sid, raw=(self._fast_raw_bank[slot] if self.fast_mode else raw.detach()), z=(self._fast_z_bank[slot] if self.fast_mode else z.detach()),
                                   birth_video=int(video_id), birth_track=str(track_key),
                                   oracle_birth_category=oracle_category,
                                   anchors=([raw.detach().cpu().numpy().astype(np.float32)] if not self.fast_mode else [np.empty((0,), dtype=np.float32)]))
                if self.fast_mode:
                    st.dispersion_tensor = torch.zeros((), dtype=raw.dtype, device=raw.device)
                self.states.append(st); candidate_index = None
                self.track_bindings[str(track_key)] = len(self.states) - 1
        if action == "EXISTING":
            if state_index is None or state_index < 0 or state_index >= len(self.states):
                action = "NEW" if len(self.states) < self.max_states else "DEFER"
                return self.apply_action(action, raw, z, video_id, track_key,
                                         oracle_category=oracle_category, quality=quality,
                                         confidence=confidence, update_allowed=update_allowed)
            st = self.states[state_index]; sid = int(st.sid)
            if oracle_category is not None and st.oracle_birth_category is not None and oracle_category != st.oracle_birth_category:
                st.impurity_count += 1
            update_prototype(st, raw, z, quality, confidence, max_anchors=self.max_anchors,
                             allow_update=update_allowed, fast_mode=self.fast_mode,
                             store_anchor_values=not self.fast_mode)
            if self.fast_mode and st.raw.device == raw.device and self._fast_raw_bank is not None:
                slot = self.states.index(st)
                self._fast_raw_bank[slot].copy_(st.raw.detach()); self._fast_z_bank[slot].copy_(st.z.detach())
                if self._fast_disp_bank is not None and st.dispersion_tensor is not None:
                    self._fast_disp_bank[slot].copy_(st.dispersion_tensor.detach())
                st.raw = self._fast_raw_bank[slot]; st.z = self._fast_z_bank[slot]
            self.track_bindings[str(track_key)] = int(state_index)
        if action in {"KNOWN", "DEFER"}:
            sid = None
        # Age all states except a freshly updated state (which update_prototype
        # already reset); this is a causal event counter, not wall-clock time.
        for st in self.states:
            if action == "EXISTING" and sid == st.sid:
                continue
            st.age += 1
        rec = {"step": int(self.step_index), "action": action, "semantic_id": sid,
               "state_index": candidate_index, "state_count": len(self.states),
               "candidate_order": [int(s.sid) for s in self.states],
               "track_key": str(track_key), "video_id": int(video_id),
               "quality": float(quality), "confidence": float(confidence),
               "states": [s.snapshot() for s in self.states] if self.record_trace else []}
        if self.record_trace:
            self.trace.append(rec)
        self.step_index += 1
        return rec


def decode_action(action_index: int, known_mask: torch.Tensor, candidate_count: int,
                  known_count: int, max_states: int) -> tuple[str, int | None]:
    """Decode padded logits after applying the episode-known mask."""
    idx = int(action_index)
    if idx < known_count:
        if bool(known_mask[idx].item()):
            return "KNOWN", idx
        return "NEW", None
    if idx < known_count + candidate_count:
        return "EXISTING", idx - known_count
    if idx == known_count + max_states:
        return "NEW", None
    return "DEFER", None


def apply_action(memory: StateMemory, action: str, raw: torch.Tensor, z: torch.Tensor,
                 video_id: int, track_key: str, **kwargs: Any) -> dict[str, Any]:
    return memory.apply_action(action, raw, z, video_id, track_key, **kwargs)


def build_candidate_tensors(memory: StateMemory, raw: torch.Tensor, video_id: int,
                            track_key: str, device: torch.device | None = None) -> dict[str, Any]:
    return memory.build_candidate_tensors(raw, video_id, track_key, device=device)
