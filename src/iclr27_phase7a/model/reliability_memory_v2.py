"""Architecture switch (Phase 7A, allowed once): rule-based evidence memory.

Same reliability-aware causal state as RACC-v1 (prototype, evidence,
maturity, provenance, per-track caps, track EMA), but the attach-or-create
decision is a closed-form evidence-scaled rule (VB-CGCD-style covariance/
evidence view + OCGCD-style reliability gating), with NO learned head.
Hyperparameters are calibrated on the legal proxy-val split only.
"""
from __future__ import annotations

import numpy as np

from src.iclr27_phase7a.model.reliability_memory import (
    MemoryState,
    TrackStats,
    compute_base_features,
    compute_known_stats,
    update_track_stats,
)


def v2_step(
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
    cfg,
):
    """One strict-causal rule decision: known / existing / new."""
    h = track_stats.ema if track_stats.ema is not None else z
    ksim, kgap = compute_known_stats(h, anchors, visible_mask)
    if ksim >= cfg["tau_k"]:
        kid = int(np.argmax(np.where(visible_mask, anchors @ h, -1.0)))
        update_track_stats(track_stats, z, score, bbox, frame_id)
        return {"decision": 0, "sid": int(known_ids[kid]),
                "slot_idx": None, "frozen_known": True}
    att, feats = mem.attach_scores_and_features(
        h, ksim, tuple(track_key), imm_penalty=cfg["pen"])
    base, rel, cons = compute_base_features(
        h, mem, ksim, kgap, att, track_stats, score, prior_hits, bbox,
        frame_id)
    n = mem.n_slots()
    if n > 0:
        ev = mem.evidence[:n]
        scaled = att * (ev / (ev + cfg["nu"]))
        j = int(np.argmax(scaled))
        best = float(scaled[j])
    else:
        j, best = -1, -1.0
    if best >= cfg["tau_attach"]:
        sid = 100000 + j
        mem._update(tuple(track_key), j, h, rel, 1.0)
        update_track_stats(track_stats, z, score, bbox, frame_id)
        return {"decision": 1, "sid": sid, "slot_idx": j,
                "frozen_known": False}
    if rel >= cfg["rel_birth"]:
        idx = mem.birth(h, track_key, rel)
        sid = 100000 + idx if idx is not None else -1
        update_track_stats(track_stats, z, score, bbox, frame_id)
        return {"decision": 2, "sid": sid, "slot_idx": idx,
                "frozen_known": False}
    # low-reliability below-threshold row: weak forced attach (avoid burst)
    if j >= 0:
        mem._update(tuple(track_key), j, h, rel, 1.0)
        sid = 100000 + j
        act = 1
    else:
        sid = int(known_ids[int(np.argmax(
            np.where(visible_mask, anchors @ h, -1.0)))])
        act = 0
    update_track_stats(track_stats, z, score, bbox, frame_id)
    return {"decision": act, "sid": sid, "slot_idx": j,
            "frozen_known": False}
