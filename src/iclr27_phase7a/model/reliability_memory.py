"""Reliability-Aware Causal Category Memory (RACC-Memory) for Phase 7A.

One unified principle: an online novel-category state is a reliability-
weighted prototype with causal evidence, maturity, birth provenance and
per-track contribution caps. Every public decision is immediate:
  KNOWN(c) / EXISTING_NOVEL(k) / NEW_NOVEL.

The learned part is a single attach-or-create head:
  - base trunk features: semantic novelty (known anchors), memory match,
    physical trajectory reliability, semantic consistency;
  - pair head scores every candidate novel slot (similarity, maturity,
    evidence, same-track provenance);
  - known/new heads score the two global options.

Memory updates are fixed and reliability-aware (not learned), so the same
code runs in training replay, Q1 dev replay and the locked heldout replay.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


def l2norm_t(x, dim=-1):
    return torch.nn.functional.normalize(x, dim=dim, eps=1e-12)


class RACCHead(nn.Module):
    """Attach-or-create decision head (the only trainable module)."""

    def __init__(self, base_dim=16, slot_dim=5, hidden=64, state_dim=32,
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
        """base: (B, base_dim); slot_feats: (B, S, slot_dim) or None.
        Returns dict(known=(B,1), new=(B,1), attach=(B,S))."""
        h = self.trunk(base)
        out = {"known": self.known_head(h), "new": self.new_head(h)}
        if slot_feats is not None and slot_feats.shape[1] > 0:
            hx = h.unsqueeze(1).expand(-1, slot_feats.shape[1], -1)
            pair_in = torch.cat([hx, slot_feats], dim=-1)
            out["attach"] = self.pair_head(pair_in).squeeze(-1)
        else:
            out["attach"] = None
        return out


@dataclass
class MemoryState:
    dim: int = 128
    max_slots: int = 500
    e_mat: float = 6.0
    imm_penalty: float = 0.10
    update_alpha: float = 0.25
    track_cap: float = 2.0
    ema_alpha: float = 0.30

    n: int = field(init=False, default=0)
    slots: np.ndarray = field(init=False, default=None)
    evidence: np.ndarray = field(init=False, default=None)
    birth_key: list = field(init=False, default_factory=list)
    birth_key_enc: np.ndarray = field(init=False, default=None)
    slot_class: np.ndarray = field(init=False, default=None)
    track_contrib: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.slots = np.zeros((self.max_slots, self.dim), dtype=np.float32)
        self.evidence = np.zeros(self.max_slots, dtype=np.float32)
        self.birth_key = []
        self.birth_key_enc = np.zeros(self.max_slots, dtype=np.int64)
        self.slot_class = np.full(self.max_slots, -1, dtype=np.int64)
        self.track_contrib = {}

    def n_slots(self):
        return int(self.n)

    def attach_scores_and_features(self, z, ksim, track_key, imm_penalty=None):
        """Returns (S,) attach scores and (S, slot_dim) features."""
        S = self.n_slots()
        if S == 0:
            return np.zeros(0, dtype=np.float32), \
                np.zeros((0, 5), dtype=np.float32)
        n = self.n
        sims = self.slots[:n] @ z
        maturity = np.minimum(1.0, self.evidence[:n] / self.e_mat)
        pen = (self.imm_penalty if imm_penalty is None else imm_penalty) \
            * (1.0 - maturity)
        att = sims - pen
        enc = track_key[0] * 1000000 + track_key[1]
        same = (self.birth_key_enc[:n] == enc).astype(np.float32)
        feats = np.stack([
            sims, maturity, np.log1p(self.evidence[:n]), same, ksim - sims,
        ], axis=1).astype(np.float32)
        return att.astype(np.float32), feats

    def _update(self, key, slot_idx, z, rel, cons):
        if slot_idx is None or slot_idx < 0 or slot_idx >= self.n:
            return
        contrib = self.track_contrib.setdefault(key, {})
        if contrib.get(slot_idx, 0.0) >= self.track_cap:
            return
        dc = min(max(rel * (0.5 + 0.5 * cons), 0.0), 1.0)
        beta = self.update_alpha * dc
        if beta > 0:
            proto = self.slots[slot_idx]
            newp = (1 - beta) * proto + beta * z
            self.slots[slot_idx] = newp / (np.linalg.norm(newp) + 1e-12)
        self.evidence[slot_idx] += 0.5 * dc
        contrib[slot_idx] = contrib.get(slot_idx, 0.0) + 0.5 * dc

    def birth(self, z, key, rel, slot_class=-1):
        """Create a slot; low-reliability births start with lower evidence."""
        if self.n >= self.max_slots:
            return None
        idx = self.n
        self.slots[idx] = z.astype(np.float32)
        ev = 0.5 + 0.5 * min(max(rel, 0.0), 1.0)
        self.evidence[idx] = ev
        self.birth_key.append(tuple(key))
        self.birth_key_enc[idx] = int(key[0]) * 1000000 + int(key[1])
        self.slot_class[idx] = int(slot_class)
        self.track_contrib.setdefault(tuple(key), {})[idx] = ev
        self.n += 1
        return idx


@dataclass
class TrackStats:
    age: int = 0
    sum_score: float = 0.0
    n: int = 0
    prev_bbox: np.ndarray = None
    prev_frame: int = -1
    stab_sum: float = 0.0
    stab_n: int = 0
    ema: np.ndarray = None


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def score_norm(v, lo=0.19, hi=0.81):
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def compute_known_stats(z, anchors, visible_mask):
    ksims = np.where(visible_mask, anchors @ z, -1.0)
    order = np.argsort(-ksims)
    ksim = float(ksims[order[0]])
    kgap = float(ksims[order[0]] - ksims[order[1]]) if anchors.shape[0] > 1 \
        else 0.0
    return ksim, kgap


def compute_base_features(z, mem, ksim, kgap, att, track_stats,
                          score, prior_hits, bbox, frame_id,
                          mask_physical=False, mask_maturity=False):
    """Causal observation features (numpy float32 vector)."""
    S = mem.n_slots()
    if S > 0:
        order_n = np.argsort(-att)
        nsim = float(att[order_n[0]])
        ngap = float(att[order_n[0]] - att[order_n[1]]) if S > 1 else 0.0
        top_ev = float(mem.evidence[order_n[0]])
        top_mat = float(min(1.0, top_ev / mem.e_mat))
    else:
        nsim, ngap, top_ev, top_mat = -1.0, 0.0, 0.0, 0.0

    age = track_stats.age
    mean_score = (track_stats.sum_score + score) / (track_stats.n + 1)
    stab = (track_stats.stab_sum / track_stats.stab_n
            if track_stats.stab_n > 0 else 0.5)
    age_norm = min(1.0, (age + 1) / 10.0)
    prior_norm = min(1.0, np.log1p(prior_hits + 1) / 3.0)
    stab_norm = min(1.0, stab / 0.8)
    rel = (0.4 * score_norm(mean_score) + 0.25 * age_norm
           + 0.20 * prior_norm + 0.15 * stab_norm)
    cons = 1.0 if track_stats.ema is None else float(np.dot(track_stats.ema, z))
    cons_norm = float(np.clip(cons, 0.0, 1.0))
    f = np.asarray([
        ksim, kgap, nsim, ngap, ksim - nsim,
        np.log1p(top_ev), top_mat,
        rel, cons_norm, np.log1p(age + 1),
        score_norm(score), np.log1p(prior_hits + 1),
        1.0 if age == 0 else 0.0,
        score_norm(mean_score), stab_norm, np.log1p(S),
    ], dtype=np.float32)
    if mask_physical:
        f[7] = 0.0
        f[10] = 0.0
        f[11] = 0.0
        f[13] = 0.0
        f[14] = 0.0
    if mask_maturity:
        f[5] = 0.0
        f[6] = 0.0
    return f, rel, cons_norm


def update_track_stats(ts, z, score, bbox, frame_id, ema_alpha=0.30):
    if ts.ema is None:
        ts.ema = z.copy()
    else:
        ts.ema = (1 - ema_alpha) * ts.ema + ema_alpha * z
        ts.ema /= (np.linalg.norm(ts.ema) + 1e-12)
    if ts.prev_bbox is not None and frame_id != ts.prev_frame:
        iou = box_iou(ts.prev_bbox, bbox)
        ts.stab_sum += iou
        ts.stab_n += 1
    ts.prev_bbox = np.asarray(bbox, dtype=np.float64)
    ts.prev_frame = int(frame_id)
    ts.sum_score += float(score)
    ts.n += 1
    ts.age += 1


def online_step(
    policy,
    z,
    mem,
    anchors,
    visible_mask,
    known_ids,
    track_stats,
    score,
    prior_hits,
    bbox,
    frame_id,
    track_key,
    slot_class=-1,
    top_k=32,
    target_slot=None,
    known_tau=None,
    use_rel=True,
    use_maturity=True,
    sem_only=False,
):
    """One strict-causal decision + memory update.

    Returns dict with logits (torch), decision, slot index, and updated
    memory. The policy forward is differentiable; memory updates are detached.
    """
    h = track_stats.ema if track_stats.ema is not None else z
    ksim, kgap = compute_known_stats(h, anchors, visible_mask)
    if known_tau is not None and ksim >= known_tau:
        kid = int(np.argmax(np.where(visible_mask, anchors @ h, -1.0)))
        update_track_stats(track_stats, z, score, bbox, frame_id)
        return {
            "logits": torch.tensor(
                [1e9, float("-inf"), float("-inf")],
                device=next(policy.parameters()).device),
            "slot_logits": None,
            "cand_idx": None,
            "decision": 0,
            "frozen_known": True,
            "slot_idx": None,
            "sid": int(known_ids[kid]),
            "rel": 1.0,
            "cons": 1.0,
        }
    att, slot_feats = mem.attach_scores_and_features(
        h, ksim, tuple(track_key),
        imm_penalty=0.0 if not use_maturity else None)
    base, rel, cons = compute_base_features(
        h, mem, ksim, kgap, att, track_stats, score, prior_hits, bbox,
        frame_id, mask_physical=not use_rel or sem_only,
        mask_maturity=not use_maturity)
    eff_rel = rel
    if sem_only:
        eff_rel = cons
    if not use_rel:
        eff_rel = 0.5
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
    # tie-break: known > existing > new
    vals = torch.stack([known_l, max_attach, new_l])
    mask = torch.zeros_like(vals)
    if attach_l is None or not attach_l.numel():
        mask[1] = -1e9
    act = int(torch.argmax(vals + mask).item())
    slot_idx = None
    if act == 1:
        pos = int(torch.argmax(attach_l).item())
        slot_idx = int(cand_idx[pos]) if cand_idx is not None else pos
    sid = -1
    if act == 0:
        kid = int(np.argmax(np.where(visible_mask, anchors @ h, -1.0)))
        sid = int(known_ids[kid])
    elif act == 1:
        sid = 100000 + slot_idx
    else:
        idx = mem.birth(h, track_key, eff_rel, slot_class)
        slot_idx = idx
        if idx is not None:
            sid = 100000 + idx
        else:
            # memory cap: fall back to best attach (or known)
            if attach_l is not None and attach_l.numel():
                act = 1
                pos = int(torch.argmax(attach_l).item())
                slot_idx = int(cand_idx[pos]) if cand_idx is not None else pos
                sid = 100000 + slot_idx
            else:
                act = 0
                kid = int(np.argmax(
                    np.where(visible_mask, anchors @ h, -1.0)))
                sid = int(known_ids[kid])
    if act == 1:
        mem._update(tuple(track_key), slot_idx, h, eff_rel, cons)
    update_track_stats(track_stats, z, score, bbox, frame_id)
    return {
        "logits": torch.stack([known_l, max_attach, new_l]),
            "slot_logits": attach_l,
            "cand_idx": cand_idx,
            "frozen_known": False,
        "known_logit": known_l,
        "new_logit": new_l,
        "attach_logits": attach_l,
        "decision": act,
        "slot_idx": slot_idx,
        "sid": sid,
        "rel": eff_rel,
        "cons": cons,
    }
