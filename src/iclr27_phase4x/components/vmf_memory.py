"""X3: vMF semantic components + sequential posterior (non-parametric).

Hypotheses at time t:
  known anchors (episode-active subset), born novel components,
  NEW component (uniform/null hypothesis), NOISE (low-quality null).
Scores are log predictive likelihoods up to shared constants; the
posterior is softmax over all hypotheses. Memory is model-in-the-loop.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class VMFSemanticMemory:
    def __init__(self, known_means: torch.Tensor, kappa: float,
                 log_prior_new: float, log_prior_noise: float,
                 noise_alpha: float = 0.0, eta: float = 0.1,
                 device: str = "cuda:0"):
        self.known = F.normalize(known_means, dim=-1).to(device)
        self.kappa = kappa
        self.log_prior_new = log_prior_new
        self.log_prior_noise = log_prior_noise
        self.noise_alpha = noise_alpha
        self.eta = eta
        self.device = device
        self.novel_means = torch.zeros(0, known_means.shape[1], device=device)
        self.novel_counts = torch.zeros(0, device=device)
        self.novel_quality = torch.zeros(0, device=device)

    def size(self):
        return int(self.novel_means.shape[0])

    def scores(self, s: torch.Tensor, q_score: float,
               known_idx: list[int] | None = None) -> tuple[torch.Tensor, dict]:
        """s: (1,256) normalized state -> (1, C+K+2) log-posterior scores."""
        kn = self.known if known_idx is None else self.known[known_idx]
        cos_k = F.normalize(s, dim=-1) @ kn.t()  # (1,C)
        logp_k = self.kappa * (cos_k - 1.0)
        if self.novel_means.shape[0] > 0:
            cos_n = F.normalize(s, dim=-1) @ self.novel_means.t()
            logp_n = self.kappa * (cos_n - 1.0)
        else:
            cos_n = torch.zeros(1, 0, device=s.device)
            logp_n = torch.zeros(1, 0, device=s.device)
        logp_new = torch.full((1, 1), self.log_prior_new, device=s.device)
        logp_noise = torch.full((1, 1),
                                self.log_prior_noise - self.noise_alpha * (q_score - 0.5),
                                device=s.device)
        scores = torch.cat([logp_k, logp_n, logp_new, logp_noise], dim=-1)
        info = {"cos_k": cos_k, "cos_n": cos_n}
        return scores, info

    def posterior(self, s: torch.Tensor, q_score: float,
                  known_idx: list[int] | None = None,
                  log_scores: torch.Tensor | None = None):
        if log_scores is None:
            log_scores, info = self.scores(s, q_score, known_idx)
        else:
            info = {}
        return torch.softmax(log_scores, dim=-1), log_scores, info

    def create(self, s: torch.Tensor, quality: float):
        s = F.normalize(s, dim=-1)
        self.novel_means = torch.cat([self.novel_means, s], dim=0)
        self.novel_counts = torch.cat([self.novel_counts,
                                       torch.ones(1, device=self.device)])
        self.novel_quality = torch.cat([self.novel_quality,
                                        torch.tensor([quality], device=self.device)])
        return self.size() - 1

    def update(self, k: int, s: torch.Tensor, quality: float):
        s = F.normalize(s, dim=-1)
        n = float(self.novel_counts[k])
        w = self.eta * quality
        mu = F.normalize((1 - w) * self.novel_means[k] + w * s[0], dim=-1)
        self.novel_means[k] = mu
        self.novel_counts[k] = n + 1
        self.novel_quality[k] = self.novel_quality[k] + quality

    def best_novel_slot(self, s: torch.Tensor) -> int:
        cos = F.normalize(s, dim=-1) @ self.novel_means.t()
        return int(cos.argmax().item())


class CompatSemanticMemory(VMFSemanticMemory):
    """X4: learned compatibility replaces kappa*(cos-1); NEW/NOISE kept as
    null hypotheses. Compatibility logits are used directly as scores."""

    def __init__(self, known_means, compat, log_prior_new, log_prior_noise,
                 noise_alpha=0.0, eta=0.1, device="cuda:0"):
        super().__init__(known_means, kappa=1.0, log_prior_new=log_prior_new,
                         log_prior_noise=log_prior_noise,
                         noise_alpha=noise_alpha, eta=eta, device=device)
        self.compat = compat
        self.compat.eval()

    def scores(self, s, q_score, known_idx=None):
        kn = self.known if known_idx is None else self.known[known_idx]
        with torch.no_grad():
            logp_k = self.compat(s, kn).unsqueeze(0)
            if self.novel_means.shape[0] > 0:
                logp_n = self.compat(s, self.novel_means).unsqueeze(0)
                cos_n = F.normalize(s, dim=-1) @ self.novel_means.t()
            else:
                logp_n = torch.zeros(1, 0, device=s.device)
                cos_n = torch.zeros(1, 0, device=s.device)
        logp_new = torch.full((1, 1), self.log_prior_new, device=s.device)
        logp_noise = torch.full((1, 1),
                                self.log_prior_noise - self.noise_alpha * (q_score - 0.5),
                                device=s.device)
        scores = torch.cat([logp_k, logp_n, logp_new, logp_noise], dim=-1)
        cos_k = F.normalize(s, dim=-1) @ kn.t()
        return scores, {"cos_k": cos_k, "cos_n": cos_n}
