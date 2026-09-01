"""Strict-causal unified category memory with immediate assign-or-create.

Action space per frame: KNOWN(c) | EXISTING_NOVEL(k) | NEW_NOVEL.
Protocol: predict with the pre-update memory, freeze the action, then update
memory only from the current observation (never retroactively).
"""
from __future__ import annotations

import torch


class CategoryMemory:
    def __init__(self, known_protos: torch.Tensor, known_ids: list[int],
                 ema_alpha: float = 0.5, update_threshold: float | None = None,
                 device="cpu"):
        self.known_protos = known_protos.to(device).float()
        self.known_ids = list(known_ids)
        self.novel_protos = torch.zeros(0, known_protos.shape[1], device=device)
        self.novel_ids: list[int] = []
        self.novel_birth_key = {}
        self.ema_alpha = ema_alpha
        self.update_threshold = update_threshold

    def reset(self):
        self.novel_protos = torch.zeros(
            0, self.known_protos.shape[1], device=self.known_protos.device)
        self.novel_ids = []
        self.novel_birth_key = {}

    @property
    def size(self) -> int:
        return len(self.novel_ids)

    def similarities(self, h: torch.Tensor) -> torch.Tensor:
        """h: (1,D) normalized -> (1, K0 + K_t) cosine similarities."""
        allp = torch.cat([self.known_protos, self.novel_protos], dim=0)
        return h @ allp.t()

    def step(self, h: torch.Tensor, tau: float,
             physical_key: tuple[int, int], update_novel: bool = True,
             update_threshold: float | None = None, allow_birth: bool = True):
        """Return (action_type, semantic_id, similarity) and apply the legal
        post-prediction memory update.

        action_type: 'known' | 'existing' | 'new'
        semantic_id: known category id, novel slot id, or new slot id.
        """
        h = torch.nn.functional.normalize(h.reshape(1, -1), dim=-1)
        sims = self.similarities(h)
        max_sim, argmax = sims.max(dim=-1)
        k0 = self.known_protos.shape[0]
        if max_sim.item() >= tau:
            idx = int(argmax.item())
            if idx < k0:
                cid = self.known_ids[idx]
                # known anchors stay frozen; no memory mutation
                return ("known", cid, float(max_sim.item()))
            slot = self.novel_ids[idx - k0]
            if update_novel and (update_threshold is None
                                 or max_sim.item() >= update_threshold):
                p = self.novel_protos[idx - k0]
                self.novel_protos[idx - k0] = torch.nn.functional.normalize(
                    (1 - self.ema_alpha) * p + self.ema_alpha * h[0], dim=-1)
            return ("existing", slot, float(max_sim.item()))
        # NEW_NOVEL birth (or forced assignment when physical birth gate
        # disallows creation; every frame still emits an immediate action)
        if not allow_birth:
            idx = int(argmax.item())
            if idx < k0:
                return ("known", self.known_ids[idx], float(max_sim.item()))
            slot = self.novel_ids[idx - k0]
            if update_novel:
                p = self.novel_protos[idx - k0]
                self.novel_protos[idx - k0] = torch.nn.functional.normalize(
                    (1 - self.ema_alpha) * p + self.ema_alpha * h[0], dim=-1)
            return ("existing", slot, float(max_sim.item()))
        slot = self.size
        self.novel_protos = torch.cat([self.novel_protos, h.clone()], dim=0)
        self.novel_ids.append(slot)
        self.novel_birth_key[slot] = physical_key
        return ("new", slot, float(max_sim.item()))
