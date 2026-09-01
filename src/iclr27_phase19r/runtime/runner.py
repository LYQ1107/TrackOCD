"""Training/inference wrapper around the Phase19R state core."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.iclr27_phase19r.runtime.state import StateMemory, decode_action


class ModelStreamController:
    def __init__(self, model: Any, *, max_states: int = 16, allow_defer: bool = True,
                 tau_ready: float = .45, tau_known: float = .20, tau_assign: float = .52):
        self.model = model
        self.max_states = int(max_states)
        self.allow_defer = bool(allow_defer)
        self.tau_ready = float(tau_ready)
        self.tau_known = float(tau_known)
        self.tau_assign = float(tau_assign)
        self.reset_stream()

    def reset_stream(self) -> None:
        self.memory = StateMemory(max_states=self.max_states, max_anchors=8)

    def _risk_decode(self, out: dict[str, torch.Tensor], bundle: dict[str, Any], quality: float,
                     known_mask: torch.Tensor) -> tuple[str, int | None, float, int]:
        return risk_decode(out, bundle, quality, known_mask, self.model.known_count,
                           self.max_states, self.allow_defer, self.tau_ready,
                           self.tau_known, self.tau_assign)

    def forward_item(self, raw: torch.Tensor, geom: torch.Tensor, quality: float,
                     video_id: int, track_key: str, known_mask: torch.Tensor,
                     allow_defer: bool | None = None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        device = raw.device
        bundle = self.memory.build_candidate_tensors(raw, int(video_id), str(track_key), device=device)
        q = torch.tensor([float(quality)], dtype=torch.float32, device=device)
        km = known_mask.to(device).bool()[None] if known_mask.ndim == 1 else known_mask.to(device).bool()
        out = self.model(raw[None], geom[None], q, km, bundle,
                         allow_defer=self.allow_defer if allow_defer is None else allow_defer)
        return out, bundle

    def process_item(self, raw: torch.Tensor, geom: torch.Tensor, quality: float,
                     video_id: int, track_key: str, known_mask: torch.Tensor,
                     *, oracle_category: int | None = None, force_action: tuple[str, int | None] | None = None,
                     confidence: float | None = None) -> dict[str, Any]:
        out, bundle = self.forward_item(raw, geom, quality, video_id, track_key, known_mask)
        if force_action is None:
            action, local_idx, conf, _ = self._risk_decode(out, bundle, quality, known_mask[None] if known_mask.ndim == 1 else known_mask)
        else:
            action, local_idx = force_action; conf = float(confidence if confidence is not None else 1.0)
        rec = self.memory.apply_action(action, raw, out["z"][0], int(video_id), str(track_key),
                                       state_index=local_idx if action == "EXISTING" else None,
                                       oracle_category=oracle_category, quality=float(quality),
                                       confidence=float(conf), update_allowed=True)
        rec.update({"action_logits": out["logits"].detach(), "candidate_score": out["candidate_score"].detach(),
                    "known_logits": out["known_logits"].detach(), "quality": float(quality),
                    "selected_confidence": float(conf), "candidate_sids": bundle["candidate_sids"]})
        # KNOWN is not an anonymous state transition, so StateMemory quite
        # correctly leaves semantic_id=None.  Preserve the selected known-slot
        # index separately for post-freeze known-safety accounting.
        rec["known_index"] = int(local_idx) if action == "KNOWN" and local_idx is not None else None
        return rec


def state_signature(memory: StateMemory) -> list[dict[str, Any]]:
    """Compact field-by-field parity signature for validation."""
    return [{"sid": int(s.sid), "count": int(s.count), "dispersion": float(s.dispersion),
             "age": int(s.age), "birth_video": int(s.birth_video), "birth_track": str(s.birth_track),
             "oracle_birth_category": s.oracle_birth_category, "impurity_count": int(s.impurity_count),
             "raw": s.raw.detach().cpu().numpy().round(7).tolist(),
             "z": s.z.detach().cpu().numpy().round(7).tolist(),
             "anchors": len(s.anchors)} for s in memory.states]


def risk_decode(out: dict[str, torch.Tensor], bundle: dict[str, Any], quality: float,
                known_mask: torch.Tensor, known_count: int, max_states: int,
                allow_defer: bool, tau_ready: float, tau_known: float,
                tau_assign: float) -> tuple[str, int | None, float, int]:
    """Shared explicit risk rule used by training and inference wrappers."""
    if allow_defer and float(quality) < float(tau_ready):
        return "DEFER", None, 1.0 - float(quality), known_count + max_states + 1
    known = out["known_logits"][0]
    km = known_mask[0].bool() if known_mask.ndim == 2 else known_mask.bool()
    if bool(km.any()):
        kj = int(torch.argmax(known.masked_fill(~km, -1e4)).item())
        ks = float(out["known_logits"][0, kj].item() / 12.0)
    else:
        kj, ks = -1, -1.0
    if kj >= 0 and ks >= tau_known:
        return "KNOWN", kj, ks, kj
    cand = out["candidate_score"][0]
    if cand.numel():
        j = int(torch.argmax(cand).item()); p = float(torch.sigmoid(cand[j]).item())
        if p >= tau_assign:
            return "EXISTING", j, p, known_count + j
    return "NEW", None, max(0.0, 1.0 - max(ks, 0.0)), known_count + max_states
