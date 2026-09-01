"""Trajectory-Conditioned Bayesian Semantic State Process (T-BSP).

Architecture A of Phase 8A. Unlike the 7A-C three-way logit heads, the model
maintains a dynamic semantic state set

    S_t = { supported-known states, online-born novel states }

and for each physical trajectory asks a single Bayesian question: which
existing state best explains the trajectory evidence, or should a new state
be spawned?

    best state = known state            -> KNOWN(c)
    best state = online-born novel state -> EXISTING_NOVEL(k)
    new-state posterior wins             -> NEW_NOVEL(k)

Each state is a Gaussian posterior over its semantic mean (mean mu, effective
count n). Each physical track maintains a causal Bayesian trajectory state
(running mean h, effective frame count w), so early evidence is uncertain
(large predictive variance) and later evidence sharpens -- this is the
trajectory-conditioning that sample-level OCD lacks. The assign-vs-spawn
decision is a posterior-predictive evidence comparison (DP-BOA-style) with a
single prior ratio rho, not a cosine threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrajectoryState:
    dim: int = 128
    mean: np.ndarray = None
    count: float = 0.0
    age: int = 0

    def update(self, z, alpha=1.0):
        """Causal Bayesian mean update: each frame adds alpha evidence."""
        if self.mean is None:
            self.mean = z.astype(np.float64).copy()
            self.count = alpha
        else:
            new = (self.mean * self.count + z.astype(np.float64) * alpha) \
                / (self.count + alpha)
            self.mean = new
            self.count += alpha
        self.age += 1

    def h(self):
        if self.mean is None:
            return None
        m = self.mean / (np.linalg.norm(self.mean) + 1e-12)
        return m.astype(np.float32)


@dataclass
class SemanticStateSet:
    dim: int = 128
    max_slots: int = 800
    sigma2: float = 0.05
    rho: float = -1.0  # spawn prior (log p_new / p_assign); calibrated

    n: int = field(init=False, default=0)
    mu: np.ndarray = field(init=False, default=None)
    count: np.ndarray = field(init=False, default=None)
    provenance: np.ndarray = field(init=False, default=None)  # 0=known, 1=novel
    birth_key: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.mu = np.zeros((self.max_slots, self.dim), dtype=np.float32)
        self.count = np.zeros(self.max_slots, dtype=np.float64)
        self.provenance = np.zeros(self.max_slots, dtype=np.int8)
        self.birth_key = np.full(self.max_slots, -1, dtype=np.int64)

    def init_known(self, known_ids, known_mu, known_counts):
        """Initialize supported-known states from legal TRAIN statistics."""
        for c, m, n in zip(known_ids, known_mu, known_counts):
            if self.n >= self.max_slots:
                break
            i = self.n
            self.mu[i] = m.astype(np.float32)
            self.count[i] = max(float(n), 1.0)
            self.provenance[i] = 0
            self.birth_key[i] = -1
            self.n += 1

    def predictive_scores(self, h, w):
        """Posterior-predictive log evidence for assigning (h,w) to each
        existing state: log N(h; mu_j, sigma2/w + sigma2/n_j) + log prior."""
        if self.n == 0:
            return np.zeros(0, dtype=np.float64)
        h = h.astype(np.float64)
        var = self.sigma2 / max(w, 1e-6) + self.sigma2 / np.maximum(
            self.count[:self.n], 1e-6)
        diff = h[None, :] - self.mu[:self.n].astype(np.float64)
        mahal2 = np.sum(diff * diff / var[:, None], axis=1)
        logdet = self.dim * np.log(2.0 * np.pi * var)
        scores = -0.5 * (logdet + mahal2)
        # log prior: uniform over existing states; rho is the log p_new ratio
        scores = scores - np.log(max(self.n, 1))
        return scores

    def spawn_score(self):
        return self.rho

    def assign(self, slot, h, w):
        if slot is None or slot < 0 or slot >= self.n:
            return
        n0 = self.count[slot]
        h64 = h.astype(np.float64)
        new_mu = (self.mu[slot].astype(np.float64) * n0 + h64 * w) / (n0 + w)
        self.mu[slot] = (new_mu / (np.linalg.norm(new_mu) + 1e-12)
                         ).astype(np.float32)
        self.count[slot] = n0 + w

    def spawn(self, h, w, key):
        if self.n >= self.max_slots:
            return None
        i = self.n
        self.mu[i] = h.astype(np.float32)
        self.count[i] = max(float(w), 1.0)
        self.provenance[i] = 1
        self.birth_key[i] = int(key[0]) * 1000000 + int(key[1])
        self.n += 1
        return i


def bsp_step(
    z,
    traj,
    states,
    known_ids,
    track_key,
    rho=None,
    sigma2=None,
    w_alpha=1.0,
):
    """One strict-causal step.

    Returns (action, sid, slot, scores). action in {0: known, 1: existing,
    2: new}. Known/existing are the SAME assign mechanism; only the
    provenance of the winning state differs.
    """
    if rho is not None:
        states.rho = rho
    if sigma2 is not None:
        states.sigma2 = sigma2
    traj.update(z, alpha=w_alpha)
    h = traj.h()
    w = traj.count
    scores = states.predictive_scores(h, w)
    spawn = states.spawn_score()
    if scores.size == 0 or float(np.max(scores)) < spawn:
        slot = states.spawn(h, w, track_key)
        if slot is not None:
            return 2, 100000 + slot, slot, None
        # memory cap: assign to best existing state
        slot = int(np.argmax(scores)) if scores.size else None
        if slot is None:
            return 0, -1, None, None
        states.assign(slot, h, w)
        prov = states.provenance[slot]
        sid = int(known_ids[slot]) if prov == 0 else 100000 + slot
        return (0 if prov == 0 else 1), sid, slot, scores
    slot = int(np.argmax(scores))
    states.assign(slot, h, w)
    prov = states.provenance[slot]
    sid = int(known_ids[slot]) if prov == 0 else 100000 + slot
    return (0 if prov == 0 else 1), sid, slot, scores
