"""TrackOCD Semantic Core: causal sequential belief + dynamic novel memory +
deferred commitment + physical-reliability conditioning.

Trainable: feature adapter, reliability-gated GRU belief, decision head
(known / existing-novel / new / defer). The novel memory itself is an online
causal mechanism (create/EMA-update/reuse with reliability gating); during
training its actions are teacher-forced to the episode GT so gradients flow
through reads and the representation, not through the model's own write
mistakes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticCore(nn.Module):
    def __init__(
        self,
        feat_dim: int = 768,
        hidden: int = 256,
        known_prototypes: torch.Tensor | None = None,  # C x 768 (normalized)
        temperature_known: float = 16.0,
        temperature_novel: float = 10.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.tau_k = temperature_known
        self.tau_n = temperature_novel
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, hidden, bias=False),
            nn.LayerNorm(hidden, elementwise_affine=False),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden + 1, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.gru = nn.GRUCell(hidden, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.new_head = nn.Sequential(
            nn.Linear(hidden + 2, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.defer_head = nn.Sequential(
            nn.Linear(hidden + 6, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        if known_prototypes is not None:
            self.register_buffer("known_raw", known_prototypes.float())
        else:
            self.register_buffer("known_raw", torch.zeros(0, feat_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 768) L2-normalized -> (..., 256) L2-normalized."""
        z = self.adapter(x)
        return F.normalize(z, dim=-1)

    def known_logits(self, h: torch.Tensor, known_idx: list[int] | None = None) -> torch.Tensor:
        """h: (B, H). Returns (B, C) cosine logits against known prototypes.
        known_idx optionally selects a subset of the canonical known matrix."""
        if self.known_raw.numel() == 0:
            return torch.zeros(h.shape[0], 0, device=h.device)
        idx = torch.tensor(known_idx, device=h.device) if known_idx is not None else None
        raw = self.known_raw if idx is None else self.known_raw[idx]
        P = F.normalize(self.adapter(raw), dim=-1)
        return self.tau_k * (h @ P.t())

    def belief_init(self, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch, self.hidden, device=device)
        m = torch.zeros(batch, self.hidden, device=device)
        return h, m

    def belief_step(
        self, z: torch.Tensor, r_phys: torch.Tensor, h: torch.Tensor,
        m: torch.Tensor, t: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reliability-gated GRU update with a residual running-mean embedding.

        z: (B,H) unit-norm; r: (B,1); t: 0-based step index.
        Returns (h_out, m_out, gate). h_out = normalize(LN(GRU) + m_out);
        m_out = running mean of z (rate 1/(t+1)). The residual keeps the
        encoder's discriminative geometry, which a bare GRU collapses.
        """
        g = torch.sigmoid(self.gate(torch.cat([z, r_phys], dim=-1)))
        h = self.ln(self.gru(g * z, h))
        m = F.normalize(m * (t / (t + 1)) + z * (1 / (t + 1)), dim=-1)
        return F.normalize(h + m, dim=-1), m, g

    def decision(
        self,
        h: torch.Tensor,
        known_idx: list[int],
        slots: "NovelMemory",
        r_phys: torch.Tensor,
        age: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Full action logits over known / existing slots / new / defer.
        Returns (logits dict, log_softmax over concatenated actions)."""
        B = h.shape[0]
        hn = F.normalize(h, dim=-1)  # similarity uses true cosine, not h-norm
        kl = self.known_logits(hn, known_idx)
        nl = slots.read(hn)  # (B, K) or zeros
        with torch.no_grad():
            max_novel = nl.max(dim=-1, keepdim=True).values if nl.shape[1] else torch.zeros(B, 1, device=h.device)
            n_slots = torch.full((B, 1), float(nl.shape[1]), device=h.device).log1p()
        # memory-conditioned birth decision: NEW must be justified relative to
        # the actual matching evidence, which transfers across category families
        new_logit = self.new_head(torch.cat([h, max_novel, n_slots], dim=-1))
        # uncertainty features for defer
        with torch.no_grad():
            k_probs = torch.softmax(kl, dim=-1)
            k_entropy = -(k_probs * torch.log(k_probs + 1e-9)).sum(-1, keepdim=True)
            if kl.shape[1] >= 2:
                top2 = torch.topk(kl, k=2, dim=-1).values
                k_margin = (top2[:, :1] - top2[:, 1:])
            else:
                k_margin = torch.zeros(B, 1, device=h.device)
            top_novel = nl.max(dim=-1, keepdim=True).values if nl.shape[1] else torch.zeros(B, 1, device=h.device)
            if nl.shape[1] >= 2:
                top2n = torch.topk(nl, k=2, dim=-1).values
                n_margin = (top2n[:, :1] - top2n[:, 1:])
            else:
                n_margin = torch.zeros(B, 1, device=h.device)
        d_in = torch.cat(
            [h, r_phys, age.clamp(max=16) / 16.0, k_entropy, k_margin, n_margin, top_novel],
            dim=-1,
        )
        defer_logit = self.defer_head(d_in)  # (B, 1)
        logits = torch.cat([kl, nl, new_logit, defer_logit], dim=-1)
        n_k = kl.shape[1]
        n_s = nl.shape[1]
        return {
            "known": kl, "novel": nl, "new": new_logit, "defer": defer_logit,
            "n_known": n_k, "n_slots": n_s,
        }, F.log_softmax(logits, dim=-1)


class NovelMemory:
    """Causal dynamic novel memory. Non-parameterized online mechanism.

    Slots: proto (256, unit-norm), support, reliability, born_frame, provenance.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.protos: torch.Tensor | None = None
        self.support: list[int] = []
        self.reliability: list[float] = []
        self.provenance: list[dict] = []

    def reset(self):
        self.protos = None
        self.support = []
        self.reliability = []
        self.provenance = []

    def size(self) -> int:
        return len(self.support)

    def read(self, h: torch.Tensor) -> torch.Tensor:
        if self.protos is None or self.protos.shape[0] == 0:
            return torch.zeros(h.shape[0], 0, device=h.device)
        sim = h @ self.protos.detach().clone().t()
        rel = torch.tensor(self.reliability, device=h.device).view(1, -1)
        return 16.0 * sim * (0.2 + 0.8 * rel.clamp(0.0, 1.0))

    def create(self, h: torch.Tensor, r_phys: float, provenance: dict) -> int:
        p = F.normalize(h.detach(), dim=-1)
        self.protos = (
            p if self.protos is None
            else torch.cat([self.protos, p], dim=0)
        )
        self.support.append(1)
        self.reliability.append(max(0.0, min(1.0, float(r_phys))))
        self.provenance.append(dict(provenance))
        return self.size() - 1

    def update(self, k: int, h: torch.Tensor, r_phys: float):
        eta = 0.35 * float(r_phys)
        p = F.normalize(h.detach(), dim=-1)
        new_p = F.normalize((1 - eta) * self.protos[k] + eta * p, dim=-1)
        self.protos[k] = new_p
        self.support[k] += 1
        self.reliability[k] = (1 - eta) * self.reliability[k] + eta * float(r_phys)

    def snapshot(self) -> dict:
        return {
            "size": self.size(),
            "support": list(self.support),
            "reliability": list(self.reliability),
            "provenance": list(self.provenance),
        }


def decision_indices(logits: dict, k: int, slot_count: int) -> str:
    """Map an action index k (over known/slots/new/defer) to an action code."""
    n_known = logits["n_known"]
    if k < n_known:
        return ("known", k)
    if k < n_known + slot_count:
        return ("existing", k - n_known)
    if k == n_known + slot_count:
        return ("new",)
    return ("defer",)
