"""Causal quality-aware trajectory semantic encoder (TSR).

Frame-level DINOv2 feature f_t + causal physical quality q_t are fused into a
hidden space; a reliability gate modulates evidence; the state is accumulated
either by a residual GRU (arch='gru') or a quality-weighted causal mean
(arch='mean'). The final state s_t is the semantic representation used by the
Phase 4U downstream hierarchical core.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSR(nn.Module):
    def __init__(
        self,
        arch: str = "gru",
        feat_dim: int = 768,
        qdim: int = 6,
        hidden: int = 256,
    ):
        super().__init__()
        assert arch in ("gru", "mean")
        self.arch = arch
        self.hidden = hidden
        self.fuse = nn.Linear(feat_dim + qdim, hidden, bias=False)
        self.ln_in = nn.LayerNorm(hidden, elementwise_affine=False)
        self.gate = nn.Sequential(
            nn.Linear(hidden + qdim, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        if arch == "gru":
            self.gru = nn.GRUCell(hidden, hidden)
            self.ln = nn.LayerNorm(hidden)

    def init_state(self, batch: int, device):
        z = torch.zeros(batch, self.hidden, device=device)
        if self.arch == "gru":
            return {"h": z, "m": z, "n": torch.zeros(batch, 1, device=device)}
        return {"num": z, "den": torch.zeros(batch, 1, device=device),
                "n": torch.zeros(batch, 1, device=device)}

    def step(self, f: torch.Tensor, q: torch.Tensor, state: dict) -> tuple[torch.Tensor, dict]:
        """f: (B,768) L2-normalized frame feature; q: (B,6) or None."""
        if q is None:
            q = torch.zeros(f.shape[0], 6, device=f.device)
        u = self.ln_in(self.fuse(torch.cat([f, q], dim=-1)))
        u = F.normalize(u, dim=-1)
        g = torch.sigmoid(self.gate(torch.cat([u, q], dim=-1)))
        if self.arch == "gru":
            n = state["n"] + 1
            h = self.ln(self.gru(g * u, state["h"]))
            m = F.normalize(
                state["m"] * (state["n"] / n) + u * (1.0 / n), dim=-1)
            s = F.normalize(h + m, dim=-1)
            return s, {"h": h, "m": m, "n": n}
        num = state["num"] + g * u
        den = state["den"] + g
        s = F.normalize(num / den.clamp(min=1e-6), dim=-1)
        return s, {"num": num, "den": den, "n": state["n"] + 1}

    def embed_sequence(
        self, feats: torch.Tensor, q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """feats: (T,768) normalized -> states (T,256) normalized."""
        T = feats.shape[0]
        state = self.init_state(1, feats.device)
        out = []
        for t in range(T):
            qt = None if q is None else q[t : t + 1]
            s, state = self.step(feats[t : t + 1], qt, state)
            out.append(s)
        return torch.cat(out, dim=0)

    def embed_batch(
        self, feats: torch.Tensor, q: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """feats: (B,T,768) normalized; mask: (B,T) bool -> (B,T,256)."""
        B, T = feats.shape[:2]
        state = self.init_state(B, feats.device)
        if q is None:
            q = torch.zeros(B, T, 6, device=feats.device)
        out = []
        for t in range(T):
            s, state = self.step(feats[:, t], q[:, t], state)
            out.append(s)
        return torch.stack(out, dim=1)
