"""Phase 4T Hierarchical TrackOCD Belief (HTB).

Level-1: fixed 3-dim open-world routing {KNOWN, NOVEL, DEFER} - independent
of the dynamic memory size K.
Level-2 (only after NOVEL): {EXISTING(k), NEW, DEFER} over the current
memory slots.

The encoder + reliability-gated residual GRU belief are reused from the
Phase 4S SemanticCore (which passed the episodic pilot); the flat variable-K
decision head is replaced by the two levels.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iclr27_phase4s.model import NovelMemory, SemanticCore


class HierarchicalCore(SemanticCore):
    def __init__(
        self,
        feat_dim: int = 768,
        hidden: int = 256,
        known_prototypes: torch.Tensor | None = None,
        qphys_dim: int = 6,
        temperature_known: float = 16.0,
        temperature_novel: float = 10.0,
        use_defer: bool = True,
        use_qphys: bool = True,
    ):
        super().__init__(feat_dim, hidden, known_prototypes,
                         temperature_known, temperature_novel)
        self.use_defer = use_defer
        self.use_qphys = use_qphys
        # reliability gate over the physical evidence vector (replaces the
        # Phase4S scalar r_phys gate when use_qphys is True)
        self.qgate = nn.Sequential(
            nn.Linear(hidden + qphys_dim, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        # Level-1: fixed 3-dim {KNOWN, NOVEL, DEFER}. The routing head sees
        # the KNOWN-branch evidence (max known logit + top-2 margin) so the
        # decision dimensionality never depends on the dynamic memory size K
        # and KNOWN vs NOVEL is identifiable.
        l1_in = hidden + qphys_dim + 1 + 2  # h, q, age, max_known, known_margin
        self.l1_head = nn.Sequential(
            nn.Linear(l1_in, 128), nn.ReLU(), nn.Linear(128, 3)
        )
        # Level-2: fixed 3-dim {EXISTING, NEW, DEFER}. EXISTING is anchored
        # directly to the best memory-read evidence (max_novel); the head
        # only produces the NEW and DEFER scores relative to that evidence.
        # h is deliberately excluded: the birth decision must be a function of
        # the memory evidence, not of appearance (otherwise the head learns a
        # constant high NEW score from the collapsed belief state and always
        # over-births).
        l2_in = qphys_dim + 1 + 3  # q, age, log1p(K), max_novel, novel_margin
        self.l2_head = nn.Sequential(
            nn.Linear(l2_in, 128), nn.ReLU(), nn.Linear(128, 2)
        )

    def qphys_feats(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """x: (B,H) normalized belief; r: (B,D) raw causal physical evidence."""
        if r.shape[1] == 1:
            # scalar fallback (synthetic episodes): expand to a constant vector
            # so the level heads always see a fixed-size evidence input
            r = r.expand(-1, 6)
        return r

    def belief_step(
        self, z: torch.Tensor, r: torch.Tensor, h: torch.Tensor,
        m: torch.Tensor, t: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """r may be (B,1) scalar (Phase4S-style) or (B,D) q_phys vector."""
        q = self.qphys_feats(z, r)
        if q.shape[1] == 6 and self.use_qphys:
            g = torch.sigmoid(self.qgate(torch.cat([z, q], dim=-1)))
        else:
            g = torch.sigmoid(self.gate(torch.cat([z, r], dim=-1)))
        h = self.ln(self.gru(g * z, h))
        m = F.normalize(m * (t / (t + 1)) + z * (1 / (t + 1)), dim=-1)
        return F.normalize(h + m, dim=-1), m, g

    def level1(
        self, h: torch.Tensor, r: torch.Tensor, age: torch.Tensor,
        kl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.qphys_feats(h, r)
        if kl.shape[1] >= 1:
            max_known = kl.max(dim=-1, keepdim=True).values
            if kl.shape[1] >= 2:
                top2 = torch.topk(kl, k=2, dim=-1).values
                known_margin = (top2[:, :1] - top2[:, 1:])
            else:
                known_margin = torch.zeros_like(max_known)
        else:
            max_known = torch.zeros(h.shape[0], 1, device=h.device)
            known_margin = torch.zeros_like(max_known)
        # Level-1 is a router over evidence, not a representation trainer:
        # detach the belief state itself so the routing head does not apply an
        # unconstrained shared direction to h, but keep the KNOWN evidence
        # differentiable: known targets pull h toward known prototypes and
        # novel targets push h away from them (the known/novel separation
        # force that the flat joint CE provided).
        h_r = h.detach()
        logits = self.l1_head(torch.cat(
            [h_r, q, age.clamp(max=16) / 16.0, max_known, known_margin], dim=-1))
        if not self.use_defer:
            logits = torch.cat([logits[:, :2], torch.full_like(logits[:, :1], -1e9)], dim=-1)
        return logits, F.log_softmax(logits, dim=-1)

    def level2(
        self, h: torch.Tensor, r: torch.Tensor, age: torch.Tensor,
        slots: NovelMemory,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        hn = F.normalize(h, dim=-1)
        nl = slots.read(hn)  # (B, K)
        q = self.qphys_feats(h, r)
        if nl.shape[1] >= 1:
            max_novel = nl.max(dim=-1, keepdim=True).values
        else:
            max_novel = torch.zeros(h.shape[0], 1, device=h.device)
        with torch.no_grad():
            n_slots = torch.full((h.shape[0], 1), float(nl.shape[1]), device=h.device).log1p()
            if nl.shape[1] >= 2:
                top2n = torch.topk(nl, k=2, dim=-1).values
                novel_margin = (top2n[:, :1] - top2n[:, 1:])
            else:
                novel_margin = torch.zeros_like(max_novel)
        l2 = self.l2_head(torch.cat(
            [q, age.clamp(max=16) / 16.0, n_slots,
             max_novel.detach(), novel_margin.detach()], dim=-1))
        if not self.use_defer:
            l2 = torch.cat([l2[:, :1], torch.full_like(l2[:, :1], -1e9)], dim=-1)
        # fixed 3-dim gate: [EXISTING(best memory), NEW, DEFER]
        logits = torch.cat([max_novel, l2], dim=-1)
        return {
            "novel": nl, "new": l2[:, :1], "defer": l2[:, 1:],
            "n_slots": nl.shape[1],
        }, F.log_softmax(logits, dim=-1)

    def decision(
        self, h: torch.Tensor, known_idx: list[int], slots: NovelMemory,
        r: torch.Tensor, age: torch.Tensor,
    ):
        kl = self.known_logits(F.normalize(h, dim=-1), known_idx)
        l1_logits, l1_lsm = self.level1(h, r, age, kl)
        l2_logits, l2_lsm = self.level2(h, r, age, slots)
        return {
            "l1": l1_logits, "l1_lsm": l1_lsm,
            "known": kl, "l2": l2_logits, "l2_lsm": l2_lsm,
        }
