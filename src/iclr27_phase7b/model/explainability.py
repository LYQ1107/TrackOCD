"""Trajectory-level Open-world Semantic Explainability (TOSE) for Phase 7B.

Unified mechanism: a single learned head performs the strict-causal
competition

    argmax { KNOWN(c), EXISTING_NOVEL(k), NEW_NOVEL }

where KNOWN evidence is *semantic explainability*: the trajectory state must
be sufficiently explained by a supported-known class-conditional Gaussian
(top-1/2 likelihood, Mahalanobis distance, cosine margin, known-space
coverage) rather than by a fixed cosine threshold. Novel memory is the
Phase 7A simple EMA baseline (counts only, no reliability reweighting in the
identity decision). Physical objectness is deliberately NOT an input of the
semantic head (Phase 7A showed proxy->Q1 spurious correlation); it is used
only for memory-update reliability via low-evidence births.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


def base_feature_names(use_dist=True, use_traj=True, traj_feats=5):
    names = []
    if use_dist:
        names += ["ksim", "kgap", "klogp", "klogp_gap", "kmahal", "kcover"]
    else:
        names += ["ksim", "kgap"]
    names += ["nsim", "ngap", "ksim_nsim", "mem_ev", "mem_mat"]
    if use_traj:
        names += ["age_norm", "cons", "is_first"]
        if traj_feats >= 5:
            names += ["flip_rate", "anchor_ent"]
    return names


class TOSEHead(nn.Module):
    """Unified known/existing/new evidence competition head."""

    def __init__(self, base_dim=14, slot_dim=5, hidden=64, state_dim=32,
                 dropout=0.1):
        super().__init__()
        self.base_dim = base_dim
        self.slot_dim = slot_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(base_dim),
            nn.Linear(base_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, state_dim),
            nn.ReLU(inplace=True),
        )
        self.known_head = nn.Linear(state_dim, 1)
        self.new_head = nn.Linear(state_dim, 1)
        self.pair_head = nn.Sequential(
            nn.LayerNorm(state_dim + slot_dim),
            nn.Linear(state_dim + slot_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, base, slot_feats=None):
        h = self.trunk(base)
        out = {"known": self.known_head(h), "new": self.new_head(h)}
        if slot_feats is not None and slot_feats.shape[1] > 0:
            hx = h.unsqueeze(1).expand(-1, slot_feats.shape[1], -1)
            pair_in = torch.cat([hx, slot_feats], dim=-1)
            out["attach"] = self.pair_head(pair_in).squeeze(-1)
        else:
            out["attach"] = None
        return out


class TOSELinearHead(nn.Module):
    """Linear evidence-competition head (architecture switch, GHOST-style).

    The KNOWN evidence is a linear combination of class-conditional
    explainability features; the NEW evidence is a linear combination of the
    same features plus trajectory uncertainty; slot selection uses a small
    MLP over novel-memory slot features. This has only ~50 learned
    parameters, so it is far less prone to proxy-OOD overfitting than the
    deep MLP head and matches the parameter-free likelihood spirit of GHOST
    and the semiparametric density-ratio view of open-set label shift.
    """

    def __init__(self, base_dim=14, slot_dim=5, hidden=16, dropout=0.1,
                 use_dist=True, use_traj=True):
        super().__init__()
        self.use_dist = use_dist
        self.use_traj = use_traj
        self.dist_n = 6 if use_dist else 2
        self.traj_n = 5 if use_traj else 0
        self.base_dim = base_dim
        self.slot_dim = slot_dim
        self.known_norm = nn.LayerNorm(self.dist_n)
        self.new_norm = nn.LayerNorm(self.dist_n + self.traj_n)
        self.known_head = nn.Linear(self.dist_n, 1)
        self.new_head = nn.Linear(self.dist_n + self.traj_n, 1)
        self.pair_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, base, slot_feats=None):
        d = base[:, :self.dist_n]
        d = self.known_norm(d)
        if self.traj_n > 0:
            t = base[:, self.dist_n + 5: self.dist_n + 5 + self.traj_n]
            new_in = torch.cat([d, t], dim=-1)
        else:
            new_in = d
        new_in = self.new_norm(new_in)
        out = {"known": self.known_head(d), "new": self.new_head(new_in)}
        if slot_feats is not None and slot_feats.shape[1] > 0:
            out["attach"] = self.pair_head(slot_feats).squeeze(-1)
        else:
            out["attach"] = None
        return out


@dataclass
class EMAMemory:
    dim: int = 128
    max_slots: int = 500
    ema_alpha: float = 0.30
    imm_penalty: float = 0.10
    maturity_scale: float = 6.0

    n: int = field(init=False, default=0)
    slots: np.ndarray = field(init=False, default=None)
    counts: np.ndarray = field(init=False, default=None)
    birth_key_enc: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.slots = np.zeros((self.max_slots, self.dim), dtype=np.float32)
        self.counts = np.zeros(self.max_slots, dtype=np.float32)
        self.birth_key_enc = np.zeros(self.max_slots, dtype=np.int64)

    def n_slots(self):
        return int(self.n)

    def attach(self, h, ksim, track_key):
        """Returns (S,) raw sims, (S,) attach scores, (S, slot_dim) feats."""
        S = self.n_slots()
        if S == 0:
            return (np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=np.float32),
                    np.zeros((0, 5), dtype=np.float32))
        sims = self.slots[:S] @ h
        maturity = np.minimum(1.0, self.counts[:S] / self.maturity_scale)
        att = sims - self.imm_penalty * (1.0 - maturity)
        enc = int(track_key[0]) * 1000000 + int(track_key[1])
        same = (self.birth_key_enc[:S] == enc).astype(np.float32)
        feats = np.stack([
            sims, np.log1p(self.counts[:S]), maturity, same, ksim - sims,
        ], axis=1).astype(np.float32)
        return sims.astype(np.float32), att.astype(np.float32), feats

    def birth(self, h, key):
        if self.n >= self.max_slots:
            return None
        idx = self.n
        self.slots[idx] = h.astype(np.float32)
        self.counts[idx] = 1.0
        self.birth_key_enc[idx] = int(key[0]) * 1000000 + int(key[1])
        self.n += 1
        return idx

    def update(self, slot_idx, h):
        if slot_idx is None or slot_idx < 0 or slot_idx >= self.n:
            return
        p = self.slots[slot_idx]
        beta = min(max(self.ema_alpha, 0.0), 1.0)
        newp = (1 - beta) * p + beta * h
        self.slots[slot_idx] = newp / (np.linalg.norm(newp) + 1e-12)
        self.counts[slot_idx] += 1.0


@dataclass
class TrackState:
    age: int = 0
    ema: np.ndarray = None
    last_anchor: int = -1
    flip_count: int = 0
    anchor_hist: np.ndarray = None

    def update_anchor(self, anchor_idx, n_anchors):
        if self.anchor_hist is None:
            self.anchor_hist = np.zeros(n_anchors, dtype=np.float32)
        if self.last_anchor >= 0 and anchor_idx != self.last_anchor:
            self.flip_count += 1
        self.last_anchor = int(anchor_idx)
        if anchor_idx >= 0:
            self.anchor_hist[anchor_idx] += 1.0

    def flip_rate(self):
        denom = max(self.age - 1, 1)
        return min(1.0, self.flip_count / denom)

    def anchor_entropy(self):
        if self.anchor_hist is None or self.anchor_hist.sum() <= 1:
            return 0.0
        p = self.anchor_hist / self.anchor_hist.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())


def top1_top2_masked(scores, mask):
    """scores: (K,) already masked with -inf. Returns (top1, top2, idx)."""
    order = np.argsort(-scores)
    k = int(mask.sum())
    if k <= 0:
        return -1.0, 0.0, -1
    i1 = int(order[0])
    v1 = float(scores[i1])
    if k >= 2:
        v2 = float(scores[order[1]])
    else:
        v2 = v1 - 0.0
    return v1, v2, i1


def build_base(h, ksim, kgap, klogp, klogp_gap, kmahal, kcover,
               sims, att, mem, track_stats, cons=0.0,
               use_dist=True, use_traj=True, traj_feats=5):
    S = mem.n_slots()
    if S > 0:
        order_n = np.argsort(-att)
        nsim = float(att[order_n[0]])
        ngap = float(att[order_n[0]] - att[order_n[1]]) if S > 1 else 0.0
        top_ev = float(np.log1p(mem.counts[order_n[0]]))
        top_mat = float(np.minimum(1.0, mem.counts[order_n[0]]
                                   / mem.maturity_scale))
    else:
        nsim, ngap, top_ev, top_mat = -1.0, 0.0, 0.0, 0.0
    age_norm = min(1.0, (track_stats.age + 1) / 10.0)
    f = []
    if use_dist:
        f += [ksim, kgap, klogp, klogp_gap, kmahal, kcover]
    else:
        f += [ksim, kgap]
    f += [nsim, ngap, ksim - nsim, top_ev, top_mat]
    if use_traj:
        f += [age_norm, cons, 1.0 if track_stats.age == 0 else 0.0]
        if traj_feats >= 5:
            f += [track_stats.flip_rate(), track_stats.anchor_entropy()]
    return np.asarray(f, dtype=np.float32)


def tose_step(
    policy,
    z,
    mem,
    anchors,
    visible_mask,
    known_ids,
    stats,
    track_stats,
    frame_id,
    track_key,
    slot_class=-1,
    top_k=32,
    target_slot=None,
    use_dist=True,
    use_traj=True,
    known_tau=None,
    asims_row=None,
    loglik_row=None,
    known_offset=0.0,
    traj_feats=5,
):
    """One strict-causal TOSE decision + memory update.

    h is the causal track-EMA state after incorporating the current frame
    (or per-frame z for the frame-level ablation). `cons` is the cosine
    between the pre-update track EMA and the current frame.
    """
    if track_stats.ema is None:
        h = z.astype(np.float32)
        cons = 0.0
    else:
        if use_traj:
            cons = float(np.dot(track_stats.ema, z))
        else:
            cons = 0.0
        if use_traj:
            h = (1 - 0.30) * track_stats.ema + 0.30 * z
            h /= (np.linalg.norm(h) + 1e-12)
        else:
            h = z.astype(np.float32)
    h = h.astype(np.float32)
    if asims_row is not None:
        asims = asims_row.astype(np.float32)
    else:
        asims = anchors @ h
    masked = np.where(visible_mask, asims, -1.0)
    ksim, kgap, _ = top1_top2_masked(masked, visible_mask)
    top_anchor = int(np.argmax(np.where(visible_mask, asims, -1.0)))
    track_stats.update_anchor(top_anchor, len(known_ids))

    # class-conditional Gaussian explainability (legal supported-known stats)
    if loglik_row is not None:
        loglik = loglik_row.astype(np.float32)
    else:
        diff = h[None, :] - stats["mu"]  # (K,D)
        mahal2 = np.sum(diff * diff / stats["sigma2"], axis=1)  # (K,)
        loglik = -0.5 * (mahal2 + stats["logdet"])
    lmask = np.where(visible_mask, loglik, -1e18)
    logp1, logp2, _ = top1_top2_masked(lmask, visible_mask)
    klogp = logp1 / h.shape[0]
    klogp_gap = (logp1 - logp2) / h.shape[0]
    if loglik_row is not None:
        diff = h[None, :] - stats["mu"]
        mahal2_row = np.sum(diff * diff / stats["sigma2"], axis=1)
        kmahal = float(np.sqrt(max(mahal2_row[np.argmax(
            np.where(visible_mask, loglik, -1e18))], 0.0))
            / np.sqrt(h.shape[0]))
    else:
        kmahal = float(np.sqrt(max(mahal2[np.argmax(
            np.where(visible_mask, loglik, -1e18))], 0.0))
            / np.sqrt(h.shape[0]))
    kcover = float(np.mean(asims[visible_mask] >= 0.55)) \
        if visible_mask.any() else 0.0

    sims, att, slot_feats = mem.attach(h, ksim, track_key)
    base = build_base(
        h, ksim, kgap, klogp, klogp_gap, kmahal, kcover,
        sims, att, mem, track_stats, cons=cons,
        use_dist=use_dist, use_traj=use_traj, traj_feats=traj_feats)

    pdev = next(policy.parameters()).device
    base_t = torch.from_numpy(base).unsqueeze(0).to(pdev)
    cand_idx = None
    if slot_feats.shape[0] > 0:
        S = slot_feats.shape[0]
        if S > top_k:
            cand_idx = np.argsort(-att)[:top_k].astype(np.int64)
            if target_slot is not None and int(target_slot) not in cand_idx:
                cand_idx = np.concatenate(
                    [cand_idx, np.asarray([int(target_slot)])])
            slot_feats = slot_feats[cand_idx]
        else:
            cand_idx = np.arange(S, dtype=np.int64)
    slot_t = torch.from_numpy(slot_feats).unsqueeze(0).to(pdev) \
        if slot_feats.shape[0] else None
    out = policy(base_t, slot_t)
    known_l = out["known"][0, 0]
    new_l = out["new"][0, 0]
    attach_l = out["attach"][0] if out["attach"] is not None else None
    max_attach = attach_l.max() if attach_l is not None and attach_l.numel() \
        else torch.full((), float("-inf"), device=known_l.device)
    vals = torch.stack([known_l, max_attach, new_l])
    if known_offset != 0.0:
        vals = vals + torch.as_tensor(
            [known_offset, 0.0, 0.0], device=vals.device)
    mask = torch.zeros_like(vals)
    if attach_l is None or not attach_l.numel():
        mask[1] = -1e9
    if known_tau is not None and ksim >= known_tau:
        act = 0
    else:
        act = int(torch.argmax(vals + mask).item())
    slot_idx = None
    if act == 1:
        pos = int(torch.argmax(attach_l).item())
        slot_idx = int(cand_idx[pos]) if cand_idx is not None else pos
    sid = -1
    if act == 0:
        kid = int(np.argmax(np.where(visible_mask, asims, -1.0)))
        sid = int(known_ids[kid])
    elif act == 1:
        sid = 100000 + slot_idx
    else:
        idx = mem.birth(h, track_key)
        slot_idx = idx
        if idx is not None:
            sid = 100000 + idx
        else:
            if attach_l is not None and attach_l.numel():
                act = 1
                pos = int(torch.argmax(attach_l).item())
                slot_idx = int(cand_idx[pos]) if cand_idx is not None else pos
                sid = 100000 + slot_idx
            else:
                act = 0
                kid = int(np.argmax(np.where(visible_mask, asims, -1.0)))
                sid = int(known_ids[kid])
    if act == 1:
        mem.update(slot_idx, h)
    track_stats.ema = h
    track_stats.age += 1
    return {
        "logits": torch.stack([known_l, max_attach, new_l]),
        "slot_logits": attach_l,
        "cand_idx": cand_idx,
        "decision": act,
        "slot_idx": slot_idx,
        "sid": sid,
        "kscore": float(torch.softmax(
            torch.stack([known_l + known_offset, max_attach, new_l]),
            dim=0)[0].item()),
        "known_logit": float(known_l.item()),
        "new_logit": float(new_l.item()),
        "attach_logits": attach_l,
        "klogp": klogp,
        "kmahal": kmahal,
        "ksim": ksim,
    }
