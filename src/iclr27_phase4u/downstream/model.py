"""Phase 4U downstream core: T3 hierarchy + memory with the pretrained TSR
representation replacing the per-frame adapter/GRU belief.

The TSR is frozen (Stage B). L1/L2 heads, qgate, gate/GRU are initialized
from the Phase 4T T3 checkpoint where shapes match; known prototypes are
rebuilt in TSR space. belief_step returns the TSR state directly so the
downstream objective cannot distort the semantic geometry.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iclr27_phase4s.episodes import load_episodic_universe
from src.iclr27_phase4t.model import HierarchicalCore
from src.iclr27_phase4u.trajectory.model import TSR


def build_tsr_known_protos(rep: TSR, device: str) -> torch.Tensor:
    """Category prototypes in TSR space from the legal train-known universe."""
    by_train, by_dev, features = load_episodic_universe()
    merged: dict[int, list[str]] = {}
    for d in (by_train, by_dev):
        for c, ids in d.items():
            merged.setdefault(c, []).extend(ids)
    protos = {}
    rep.eval()
    with torch.no_grad():
        for c in sorted(merged):
            es = []
            for sid in merged[c]:
                f = torch.from_numpy(features[sid]).to(device)
                st = rep.embed_sequence(f, None)
                es.append(st[-1].cpu().numpy().astype(np.float32))
            p = np.mean(np.stack(es), axis=0)
            p = p / (np.linalg.norm(p) + 1e-12)
            protos[c] = p
    arr = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    return torch.from_numpy(arr)


class HierarchicalTSRCore(HierarchicalCore):
    def __init__(
        self,
        rep: TSR,
        known_protos: torch.Tensor,
        use_defer: bool = False,
        use_qphys: bool = True,
        freeze_rep: bool = True,
    ):
        super().__init__(768, 256, known_prototypes=None,
                         use_defer=use_defer, use_qphys=use_qphys)
        self.rep = rep
        self.freeze_rep = freeze_rep
        if freeze_rep:
            for p in rep.parameters():
                p.requires_grad_(False)
        self.register_buffer("known_raw", known_protos.float())
        self._states = None
        self._ptr = 0

    def load_t3_init(self, checkpoint: str, device: str):
        ck = torch.load(checkpoint, map_location=device)
        sd = dict(ck["model"])
        sd.pop("known_raw", None)
        sd.pop("adapter.0.weight", None)
        sd.pop("adapter.1.weight", None)
        sd.pop("adapter.1.bias", None)
        self.load_state_dict(sd, strict=False)

    def known_logits(self, h: torch.Tensor, known_idx: list[int] | None = None):
        raw = self.known_raw if known_idx is None else self.known_raw[known_idx]
        return self.tau_k * (h @ raw.t())

    def begin_occurrence(self, feats: torch.Tensor, q: torch.Tensor | None = None):
        if self.freeze_rep:
            with torch.no_grad():
                self._states = self.rep.embed_sequence(feats, q)
        else:
            self._states = self.rep.embed_sequence(feats, q)
        self._ptr = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        s = self._states[self._ptr]
        self._ptr += 1
        return s.unsqueeze(0)

    def belief_step(self, z, r, h, m, t):
        # representation is already the causal semantic state; keep h=z
        return z, m, torch.ones(z.shape[0], 1, device=z.device)
