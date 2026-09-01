"""Dual-space evidence for Phase 4V.

Known branch  : R3 TSR (Phase4U Stage A) + its 48-way linear classifier.
                Frozen; produces known-class logits for recognition.
Novel branch  : Stage C TSR (joint-tuned) + Stage C L2 head + dynamic
                NovelMemory. Frozen; produces existing/new evidence and
                memory read logits for novel semantic identity.

The independent router consumes summaries of both branches (never raw
per-class logits except through summary statistics).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.iclr27_phase4s.model import NovelMemory
from src.iclr27_phase4u.trajectory.model import TSR

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def load_known_branch(device: str, checkpoint: str | None = None):
    """R3 TSR + 48-way classifier. Returns (tsr, cls_head)."""
    ck = torch.load(ROOT / (checkpoint or
                            "outputs/iclr27_phase4u/representation/r3_mixed_gru/checkpoint.pth"),
                    map_location=device)
    tsr = TSR(arch=ck.get("arch", "gru")).to(device)
    tsr.load_state_dict(ck["model"])
    tsr.eval()
    cls = torch.nn.Linear(256, 48).to(device)
    cls.load_state_dict(ck["cls"])
    cls.eval()
    return tsr, cls


def load_novel_branch(device: str, checkpoint: str | None = None):
    """Stage C TSR + L2 head (existing/new evidence). Returns (tsr, l2)."""
    ck = torch.load(ROOT / (checkpoint or
                            "outputs/iclr27_phase4u/downstream/d2_joint_v2/checkpoint.pth"),
                    map_location=device)
    sd = ck["model"]
    rep_sd = {k[len("rep."):]: v for k, v in sd.items() if k.startswith("rep.")}
    tsr = TSR(arch="gru").to(device)
    tsr.load_state_dict(rep_sd)
    tsr.eval()
    l2 = torch.nn.Sequential(
        torch.nn.Linear(6 + 1 + 3, 128), torch.nn.ReLU(), torch.nn.Linear(128, 2)
    ).to(device)
    l2_sd = {k[len("l2_head."):]: v for k, v in sd.items() if k.startswith("l2_head.")}
    # l2_head = Sequential(Linear(10,128), ReLU, Linear(128,2)); same structure
    l2.load_state_dict(l2_sd)
    l2.eval()
    return tsr, l2


def known_evidence(kl: torch.Tensor, idx: list[int] | None = None) -> np.ndarray:
    """kl: (1,48) raw logits -> [top1_p, margin_p, entropy, energy].

    If idx is given, evidence is computed over that subset of the known
    vocabulary (episode-known classes), which simulates an OOV novel whose
    true category is absent from the current vocabulary.
    """
    kl = kl if idx is None else kl[:, idx]
    p = F.softmax(kl, dim=-1)
    top2 = torch.topk(p, k=2, dim=-1).values
    top1 = top2[:, :1]
    margin = (top2[:, :1] - top2[:, 1:])
    entropy = -(p * torch.log(p + 1e-9)).sum(-1, keepdim=True)
    energy = torch.logsumexp(kl, dim=-1, keepdim=True)
    return torch.cat([top1, margin, entropy, energy], dim=-1).detach().cpu().numpy()[0]


def novel_evidence(nl: torch.Tensor, l2_new: torch.Tensor,
                   q: torch.Tensor, age: float, K: int) -> np.ndarray:
    """-> [max_novel, novel_margin, new_logit, log1p(K)]."""
    if nl.shape[1] >= 1:
        max_n = nl.max(dim=-1, keepdim=True).values
        if nl.shape[1] >= 2:
            top2n = torch.topk(nl, k=2, dim=-1).values
            margin_n = (top2n[:, :1] - top2n[:, 1:])
        else:
            margin_n = torch.zeros_like(max_n)
    else:
        max_n = torch.zeros(1, 1, device=nl.device)
        margin_n = torch.zeros_like(max_n)
    new_logit = l2_new.detach()
    kk = torch.full((1, 1), float(K), device=nl.device).log1p()
    return torch.cat([max_n, margin_n, new_logit, kk],
                     dim=-1).detach().cpu().numpy()[0]


def proto_evidence(s: torch.Tensor, protos: torch.Tensor,
                   idx: list[int] | None = None, tau: float = 0.1
                   ) -> np.ndarray:
    """Prototype-similarity known evidence over the current known set.

    Returns [top1_sim, margin_sim, entropy_sim, energy_sim] with softmax
    temperature tau. If idx is given, only those prototypes are used.
    """
    p = torch.nn.functional.normalize(s, dim=-1)
    sims = p @ protos[idx].t() if idx is not None else p @ protos.t()
    logits = sims / tau
    ps = torch.softmax(logits, dim=-1)
    top2 = torch.topk(ps, k=min(2, ps.shape[-1]), dim=-1).values
    top1 = top2[:, :1]
    margin = top1 - top2[:, 1:] if ps.shape[-1] >= 2 else torch.zeros_like(top1)
    entropy = -(ps * torch.log(ps + 1e-9)).sum(-1, keepdim=True)
    energy = torch.logsumexp(logits, dim=-1, keepdim=True)
    return torch.cat([top1, margin, entropy, energy],
                     dim=-1).detach().cpu().numpy()[0]


def router_vector(known_ev: np.ndarray, known_full_ev: np.ndarray,
                  novel_ev: np.ndarray,
                  q: np.ndarray) -> np.ndarray:
    """vocab-known(4) + full-known(4) + novel(4) + disagreement(1) + q(6) = 19 dims."""
    disagreement = np.array([known_full_ev[3] - novel_ev[0]])  # energy - max_memory
    return np.concatenate([known_ev, known_full_ev, novel_ev,
                           disagreement, q]).astype(np.float32)


class DualSpaceStep:
    """Incremental dual-space state for one track prefix."""

    def __init__(self, known_tsr, known_cls, novel_tsr, l2_head, device):
        self.known_tsr = known_tsr
        self.known_cls = known_cls
        self.novel_tsr = novel_tsr
        self.l2_head = l2_head
        self.device = device
        self.k_state = known_tsr.init_state(1, device)
        self.n_state = novel_tsr.init_state(1, device)

    def step(self, f: torch.Tensor, q_real: torch.Tensor,
             r_scalar: float, age: float,
             memory: NovelMemory,
             known_idx: list[int] | None = None) -> tuple[np.ndarray, torch.Tensor,
                                          torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            s_k, self.k_state = self.known_tsr.step(f, q_real, self.k_state)
            s_n, self.n_state = self.novel_tsr.step(f, q_real, self.n_state)
            kl = self.known_cls(s_k)
            nl = memory.read(s_n) if memory.size() > 0 else torch.zeros(1, 0, device=self.device)
            K = memory.size()
            if nl.shape[1] >= 1:
                max_n = nl.max(dim=-1, keepdim=True).values
                top2n = torch.topk(nl, k=2, dim=-1).values if nl.shape[1] >= 2 else None
                margin_n = (top2n[:, :1] - top2n[:, 1:]) if top2n is not None \
                    else torch.zeros_like(max_n)
            else:
                max_n = torch.zeros(1, 1, device=self.device)
                margin_n = torch.zeros_like(max_n)
            n_slots = torch.full((1, 1), float(K), device=self.device).log1p()
            q_l2 = torch.full((1, 6), float(r_scalar), device=self.device)
            l2_in = torch.cat([q_l2, torch.tensor([[min(age, 16) / 16.0]],
                                                  device=self.device), n_slots,
                               max_n.detach(), margin_n.detach()], dim=-1)
            l2 = self.l2_head(l2_in)
            l2_new = l2[:, :1]
        ke = known_evidence(kl, known_idx)
        ke_full = known_evidence(kl)
        ne = novel_evidence(nl, l2_new, q_real, age, K)
        return (router_vector(ke, ke_full, ne, q_real.cpu().numpy()[0]),
                s_k, s_n, nl, l2_new)


def build_known_protos(known_tsr, device):
    """Category prototypes in known (R3) space from the episodic universe."""
    from src.iclr27_phase4s.episodes import load_episodic_universe
    by_train, by_dev, features = load_episodic_universe()
    merged = {}
    for d in (by_train, by_dev):
        for c, ids in d.items():
            merged.setdefault(c, []).extend(ids)
    protos = {}
    with torch.no_grad():
        for c in sorted(merged):
            es = []
            for sid in merged[c]:
                f = torch.from_numpy(features[sid]).to(device)
                st = known_tsr.embed_sequence(f, None)
                es.append(st[-1].cpu().numpy().astype(np.float32))
            p = np.mean(np.stack(es), axis=0)
            p = p / (np.linalg.norm(p) + 1e-12)
            protos[c] = p
    arr = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    return torch.from_numpy(arr)
