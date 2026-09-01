"""Phase 4Y ADSSI: amortized dynamic semantic-state inference.

Components:
  - ObservationEncoder: Stage C TSR state -> z (trainable projection).
  - StateSet: known states (init from anchors) + born novel states +
    trajectory-conditioned NEW proposal; permutation-invariant attention.
  - Assignment scores: set-conditioned (self-attention over states +
    cross-attention context + pairwise term).
  - Transition: learned gated residual update.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationEncoder(nn.Module):
    def __init__(self, in_dim=256, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(),
                                 nn.LayerNorm(d))

    def forward(self, s):
        return self.net(s)


class StateAttention(nn.Module):
    """One-layer self-attention over state tokens + cross-attention query."""

    def __init__(self, d=128, heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, states, z):
        # states: (1, M, d); z: (1, 1, d)
        h, _ = self.self_attn(states, states, states)
        h = self.ln1(states + h)
        c, _ = self.cross_attn(z, h, h)
        c = self.ln2(z + c)
        return h, c


class ADSSI(nn.Module):
    def __init__(self, in_dim=256, d=128, heads=4, anchor_proj=None):
        super().__init__()
        self.d = d
        self.obs = ObservationEncoder(in_dim, d)
        self.anchor_proj = nn.Linear(in_dim, d, bias=False)
        self.attn = StateAttention(d, heads)
        self.score_mlp = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(),
                                       nn.Linear(d, 1))
        self.proposal = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(),
                                      nn.Linear(d, d))
        self.update_gate = nn.Linear(3 * d, d)
        self.update_cell = nn.Linear(3 * d, d)

    def init_known(self, anchors: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.anchor_proj(anchors), dim=-1)

    def propose_new(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proposal(torch.cat([z, context], dim=-1)), dim=-1)

    def scores(self, z, states, context):
        """z (1,1,d); states (1,M,d); context (1,1,d) -> (1,M)."""
        zs = z.expand(-1, states.shape[1], -1)
        cs = context.expand(-1, states.shape[1], -1)
        x = torch.cat([zs, states, zs * states], dim=-1)
        raw = self.score_mlp(x).squeeze(-1)  # (1,M)
        return raw

    def transition(self, h, z, q):
        """h (1,d); z (1,1,d); q scalar tensor -> updated h (1,d)."""
        zz = z[:, 0]
        x = torch.cat([h, zz, torch.full_like(zz, float(q))], dim=-1)
        g = torch.sigmoid(self.update_gate(x))
        u = torch.tanh(self.update_cell(x))
        return F.normalize(h + g * u, dim=-1)

    def forward_state_set(self, known: torch.Tensor, novel: torch.Tensor,
                          z: torch.Tensor, q: float):
        """known (C,d); novel (K,d); z (1,d) -> (scores, new_proposal,
        contextualized_states, context)."""
        z1 = z.unsqueeze(1)
        proposal = self.propose_new(z, z)  # (1,d)
        states = torch.cat([known, novel, proposal], dim=0).unsqueeze(0)
        h_ctx, c = self.attn(states, z1)
        scores = self.scores(z1, h_ctx, c)[0]
        proposal_final = self.propose_new(z, c[:, 0])
        return scores, proposal_final, h_ctx[0], c[0]


class DynamicStateMemory:
    """Online state set with known anchors + born novel states."""

    def __init__(self, model: ADSSI, anchors: torch.Tensor, device: str):
        self.model = model
        with torch.no_grad():
            self.known = model.init_known(anchors)  # (C,d)
        self.novel = torch.zeros(0, model.d, device=device)
        self.quality = torch.zeros(0, device=device)
        self.counts = torch.zeros(0, device=device)
        self.device = device

    def size(self):
        return int(self.novel.shape[0])

    def create(self, h: torch.Tensor, q: float):
        self.novel = torch.cat([self.novel, h.detach()], dim=0)
        self.quality = torch.cat([self.quality, torch.tensor([q], device=self.device)])
        self.counts = torch.cat([self.counts, torch.ones(1, device=self.device)])
        return self.size() - 1

    def update(self, k: int, z: torch.Tensor, q: float):
        h = self.model.transition(self.novel[k:k + 1], z.unsqueeze(0), q)
        self.novel[k] = h.detach()
        self.counts[k] = self.counts[k] + 1
        self.quality[k] = self.quality[k] + q

    def infer(self, z: torch.Tensor, q: float):
        """Returns (scores over C+K+1, proposal vector, info)."""
        scores, prop, h_ctx, c = self.model.forward_state_set(
            self.known, self.novel, z, q)
        return scores, prop, h_ctx, c
