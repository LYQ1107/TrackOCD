"""Small category-level adapters and causal weights, not ReID/physical MOT.

Supervision belongs to the external fit loop. Forward takes only descriptors,
quality and observed elapsed frames; never class, identity, role or future.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SemanticAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Linear(512, 256))

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(visual), dim=-1)


class CategoryEvidence(nn.Module):
    def __init__(self, learned_evidence: bool):
        super().__init__()
        self.adapter = SemanticAdapter()
        self.learned_evidence = learned_evidence
        if learned_evidence:
            self.reliability = nn.Sequential(nn.Linear(261, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, visual: torch.Tensor, quality: torch.Tensor, elapsed: torch.Tensor) -> dict:
        # Shapes are [batch, visible observations, 768] and matching scalars.
        semantic = self.adapter(visual)
        count = torch.arange(1, visual.shape[1] + 1, device=visual.device, dtype=visual.dtype)[None, :, None]
        cumulative = semantic.cumsum(dim=1) / count
        consistency = (semantic * F.normalize(cumulative, dim=-1)).sum(-1, keepdim=True)
        dispersion = (1 - cumulative.norm(dim=-1, keepdim=True)).clamp(0, 1)
        if self.learned_evidence:
            count_feature = count.expand(visual.shape[0], -1, -1).log1p() / 4
            elapsed_feature = elapsed[..., None].clamp(min=0).log1p() / 10
            cues = torch.cat((semantic, quality[..., None], consistency, dispersion, count_feature, elapsed_feature), dim=-1)
            logits = self.reliability(cues).squeeze(-1)
            weights = torch.sigmoid(logits) * quality.clamp(min=0)
        else:
            logits = None
            weights = quality.clamp(min=0)
        weights = weights.clamp(min=1e-6)
        normalized_weights = weights / weights.sum(dim=1, keepdim=True)
        evidence = F.normalize((semantic * normalized_weights[..., None]).sum(dim=1), dim=-1)
        maturity = 1 / normalized_weights.square().sum(dim=1)
        return {"embedding": evidence, "weights": weights, "reliability_logits": logits,
                "effective_maturity": maturity,
                "uncertainty": (1 - (semantic * evidence[:, None]).sum(-1).mul(normalized_weights).sum(1)).clamp(0, 1)}


def cross_video_category_loss(embedding: torch.Tensor, categories: torch.Tensor,
                              videos: torch.Tensor, physical_ids: torch.Tensor,
                              temperature: float = .1) -> torch.Tensor:
    """Same-category cross-video individuals only; labels never go to forward."""
    same_category = categories[:, None] == categories[None, :]
    same_video = videos[:, None] == videos[None, :]
    same_physical = same_video & (physical_ids[:, None] == physical_ids[None, :])
    positive = same_category & ~same_video & ~same_physical
    negative = ~same_category
    allowed = positive | negative
    if not positive.any(dim=1).all() or not negative.any(dim=1).all():
        raise ValueError("Every fit anchor needs cross-video category positives and category negatives")
    scores = embedding @ embedding.T / temperature
    log_probability = scores - torch.logsumexp(scores.masked_fill(~allowed, -torch.inf), dim=1, keepdim=True)
    return -(log_probability.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()
