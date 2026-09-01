"""Learned immediate assign-or-create head.

Decision features at frame t (strict-causal, pre-update memory):
  max_known_sim, max_novel_sim (0 when no novel slots), all-space margin,
  normalized track age, log1p(K) (current number of novel states).

Output: 2-way [ASSIGN, NEW]. ASSIGN -> argmax prototype in the unified
semantic space (known or existing novel); NEW -> instantiate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CreationHead(nn.Module):
    def __init__(self, feat_dim: int = 5, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def decision_features(h: torch.Tensor, memory, age: float,
                      max_age: float = 12.0) -> torch.Tensor:
    """h: (1,D) normalized state; memory: CategoryMemory (pre-update)."""
    h = F.normalize(h.reshape(1, -1), dim=-1)
    sims = memory.similarities(h)[0]  # (K0 + K_t,)
    k0 = memory.known_protos.shape[0]
    known_part = sims[:k0]
    novel_part = sims[k0:]
    max_known = known_part.max().item() if k0 > 0 else 0.0
    max_novel = novel_part.max().item() if novel_part.numel() > 0 else 0.0
    if sims.numel() >= 2:
        top2 = torch.topk(sims, 2).values
        margin = float(top2[0] - top2[1])
    else:
        margin = 0.0
    x = torch.tensor([
        max_known, max_novel, margin, min(age, max_age) / max_age,
        float(torch.log1p(torch.tensor(memory.size, dtype=torch.float32))),
    ], dtype=torch.float32)
    return x


def head_action(head: CreationHead, h: torch.Tensor, memory,
                age: float) -> tuple[str, int, float]:
    with torch.no_grad():
        x = decision_features(h, memory, age).unsqueeze(0).to(h.device)
        logits = head(x)
        p = torch.softmax(logits, dim=-1)[0]
    if int(p.argmax()) == 1:
        return ("new", memory.size, float(p[1].item()))
    hh = F.normalize(h.reshape(1, -1), dim=-1)
    sims = memory.similarities(hh)[0]
    k0 = memory.known_protos.shape[0]
    idx = int(sims.argmax().item())
    if idx < k0:
        return ("known", memory.known_ids[idx], float(sims[idx].item()))
    return ("existing", memory.novel_ids[idx - k0], float(sims[idx].item()))
