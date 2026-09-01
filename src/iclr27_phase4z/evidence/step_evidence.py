"""Shared per-step Phase 4Z routing evidence (frozen O1c model)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def stats_of(kl: torch.Tensor) -> np.ndarray:
    """[top1_p, margin, entropy, energy, max_kl] over known logits."""
    p = F.softmax(kl, dim=-1)
    top2 = torch.topk(p, k=min(2, p.shape[-1]), dim=-1).values
    top1 = top2[:, :1]
    margin = (top1 - top2[:, 1:]) if p.shape[-1] >= 2 else torch.zeros_like(top1)
    entropy = -(p * torch.log(p + 1e-9)).sum(-1, keepdim=True)
    energy = torch.logsumexp(kl, dim=-1, keepdim=True)
    max_kl = kl.max(dim=-1, keepdim=True).values
    return torch.cat([top1, margin, entropy, energy, max_kl],
                     dim=-1).detach().cpu().numpy()[0]


def proto_stats(h: torch.Tensor, protos: torch.Tensor, idx,
                tau: float = 0.1) -> np.ndarray:
    """[top1_sim, margin_sim, entropy_sim, energy_sim]."""
    h = F.normalize(h, dim=-1)
    sims = h @ protos[idx].t()
    logits = sims / tau
    ps = F.softmax(logits, dim=-1)
    top2 = torch.topk(ps, k=min(2, ps.shape[-1]), dim=-1).values
    top1 = top2[:, :1]
    margin = (top1 - top2[:, 1:]) if ps.shape[-1] >= 2 else torch.zeros_like(top1)
    entropy = -(ps * torch.log(ps + 1e-9)).sum(-1, keepdim=True)
    energy = torch.logsumexp(logits, dim=-1, keepdim=True)
    return torch.cat([top1, margin, entropy, energy],
                     dim=-1).detach().cpu().numpy()[0]


def step_evidence(model, zt, qt, age, q_np, r_scalar, active_idx, full_idx):
    """Returns (ev 30-d float32, l1p 3-d float32, kl_full (1,48) tensor)."""
    with torch.no_grad():
        kl_a = model.known_logits(zt, active_idx)
        kl_f = model.known_logits(zt, full_idx)
        _, l1_lsm = model.level1(zt, qt, age, kl_a)
        l1p = torch.exp(l1_lsm[0]).cpu().numpy().astype(np.float32)
        pa = proto_stats(zt, model.known_raw, active_idx, tau=0.1)
        pf = proto_stats(zt, model.known_raw, full_idx, tau=0.1)
        resid = float(1.0 - max(pa[0], 0.0))
        ev = np.concatenate([
            pa, pf, stats_of(kl_a), stats_of(kl_f), l1p,
            np.array([resid], np.float32),
            np.asarray(q_np, dtype=np.float32),
            np.array([r_scalar], np.float32),
            np.array([min(float(age[0, 0]), 16.0) / 16.0], np.float32),
        ]).astype(np.float32)
    return ev, l1p, kl_f
