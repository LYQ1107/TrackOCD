"""Causal trajectory semantic adapter (Architecture A representation).

The adapter consumes the frozen corrected per-frame semantic feature z_t and
the physical track's own causal GRU state, and outputs the trajectory
representation h_t that feeds the Bayesian semantic state process.  The
physical identity is only used to select which GRU state is updated; the
semantic state set is shared across all physical tracks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTrajectoryAdapter(nn.Module):
    def __init__(self, dim=128, hidden=128, rho_init=40.0, sigma2=0.05,
                 layers=1, frame_level=False):
        super().__init__()
        self.dim = dim
        self.frame_level = frame_level
        self.gru = nn.GRU(dim, hidden, num_layers=layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        # small-gain init: near identity with the input feature at start
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.gru.weight_ih_l0, gain=0.5)
        nn.init.orthogonal_(self.gru.weight_hh_l0, gain=0.5)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)
        self.sigma2 = float(sigma2)
        self.rho = nn.Parameter(torch.tensor(float(rho_init)))

    def forward(self, z, state=None):
        """z: (B, dim); state: (layers, B, hidden). Returns (B, dim), state."""
        if self.frame_level:
            raw = self.mlp(z) + z
        else:
            out, state = self.gru(z.unsqueeze(1), state)
            raw = self.mlp(out[:, 0]) + z
        self.last_raw = raw
        out = F.normalize(raw, dim=-1)
        return out, state

    def new_state(self, batch=1):
        dev = next(self.parameters()).device
        return torch.zeros(self.gru.num_layers, batch, self.gru.hidden_size,
                           device=dev)


class TorchSemanticStateSet(nn.Module):
    """Differentiable Gaussian semantic state set used in episodic training.

    Each state is a Gaussian posterior over its semantic mean (mu, count).
    The predictive log score of trajectory evidence (h, w) against state j is

        log N(h; mu_j, sigma2/w + sigma2/n_j) - log(n_states)

    and the spawn option is a single trainable log prior rho.  State
    statistics are updated online with detached values; only h carries
    gradients into the adapter.
    """

    def __init__(self, dim=128, max_slots=512, sigma2=0.05,
                 score_mode="gaussian", no_evidence=False, cosine_temp=20.0,
                 no_update=False):
        super().__init__()
        self.dim = dim
        self.max_slots = max_slots
        self.sigma2 = float(sigma2)
        self.score_mode = score_mode
        self.no_evidence = no_evidence
        self.cosine_temp = float(cosine_temp)
        self.no_update = no_update
        self.register_buffer("mu", torch.zeros(max_slots, dim))
        self.register_buffer("count", torch.zeros(max_slots))
        self.register_buffer("provenance", torch.zeros(max_slots,
                                                       dtype=torch.long))
        self.n = 0

    def reset(self):
        self.n = 0
        self.mu.zero_()
        self.count.zero_()
        self.provenance.zero_()

    def init_known(self, mu, count):
        """mu: (K, dim) tensor on device; count: (K,)."""
        k = mu.shape[0]
        self.mu[:k] = mu
        self.count[:k] = count
        self.provenance[:k] = 0
        self.n = int(k)

    def log_scores(self, h, w):
        n = self.n
        if n == 0:
            return torch.zeros(0, device=h.device)
        if self.score_mode == "cosine":
            scores = self.cosine_temp * (
                h[None, :] * self.mu[:n]).sum(dim=1)
        else:
            if self.no_evidence:
                var = torch.full(
                    (n,), self.sigma2, device=h.device, dtype=h.dtype)
            else:
                var = self.sigma2 / max(float(w), 1e-6) + \
                    self.sigma2 / torch.clamp(self.count[:n], min=1e-6)
            diff = h[None, :] - self.mu[:n]
            mahal2 = (diff * diff / var[:, None]).sum(dim=1)
            logdet = self.dim * torch.log(2.0 * torch.pi * var)
            scores = -0.5 * (logdet + mahal2)
        scores = scores - torch.log(torch.tensor(float(n), device=h.device))
        return scores

    def logits(self, h, w, rho):
        scores = self.log_scores(h, w)
        return torch.cat([scores, rho.reshape(1)], dim=0)

    def assign(self, slot, h, w):
        if self.no_update:
            return
        h = h.detach()
        mu = self.mu.clone()
        count = self.count.clone()
        n0 = count[slot]
        new_mu = (mu[slot] * n0 + h * float(w)) / (n0 + float(w))
        mu[slot] = F.normalize(new_mu, dim=-1)
        count[slot] = n0 + float(w)
        self.mu = mu
        self.count = count

    def spawn(self, h, w):
        h = h.detach()
        if self.n >= self.max_slots:
            return None
        i = self.n
        mu = self.mu.clone()
        count = self.count.clone()
        prov = self.provenance.clone()
        mu[i] = F.normalize(h, dim=-1)
        count[i] = max(float(w), 1.0)
        prov[i] = 1
        self.mu = mu
        self.count = count
        self.provenance = prov
        self.n += 1
        return i

    def decide(self, h, w, rho):
        """Numpy-free online decision: (action, sid, slot)."""
        logits = self.logits(h.detach(), w, rho.detach())
        idx = int(torch.argmax(logits))
        n = self.n
        if idx == n:
            slot = self.spawn(h, w)
            return (2, 100000 + slot if slot is not None else -1, slot)
        slot = idx
        prov = int(self.provenance[slot])
        self.assign(slot, h, w)
        return (0 if prov == 0 else 1, slot if prov == 0 else 100000 + slot,
                slot)


def normalize_rows(x):
    return F.normalize(x, dim=-1)
