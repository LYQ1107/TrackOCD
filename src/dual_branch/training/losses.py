from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_loss(z, y, tau=0.1):
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / tau
    n = z.shape[0]
    mask = (y.unsqueeze(0) == y.unsqueeze(1)) & ~torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim - torch.eye(n, device=z.device) * 1e9
    denom = torch.logsumexp(sim, dim=1, keepdim=True)
    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * mask.float()).sum(1)
    log_prob = torch.log(pos_sum / torch.exp(denom).squeeze(1) + 1e-9)
    return -log_prob.mean()


def classification_loss(logits, y):
    return F.cross_entropy(logits, y)
