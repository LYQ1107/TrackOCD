"""A frozen-feature semantic adapter for TrackOCD v2.

The physical frontend produces a category-agnostic visual track feature.  The
adapter has two deliberately separate outputs:

* ``z_instance`` is the input feature itself and is never replaced by the
  semantic branch;
* ``z_semantic`` is a small learned projection used only for cross-track
  category correspondence and OCD routing.

The module contains no category, video, physical-ID, text, future, or
controller input.  Labels and grouping metadata are accepted only by the
standalone TRAIN loss functions below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class SemanticAdapterConfig:
    """Architecture parameters for the semantic-only branch."""

    input_dim: int = 768
    semantic_dim: int = 256
    hidden_dim: int = 256
    temperature: float = 0.07
    use_residual: bool = True


class SemanticAdapter(nn.Module):
    """Map frozen visual track features to a separate semantic space."""

    def __init__(self, config: SemanticAdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or SemanticAdapterConfig()
        if self.config.input_dim < 1 or self.config.semantic_dim < 1 or self.config.hidden_dim < 1:
            raise ValueError("adapter dimensions must be positive")
        if self.config.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.projection = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.semantic_dim),
        )
        self.residual = (
            nn.Linear(self.config.input_dim, self.config.semantic_dim, bias=False)
            if self.config.use_residual
            else None
        )

    @property
    def input_dim(self) -> int:
        return int(self.config.input_dim)

    @property
    def semantic_dim(self) -> int:
        return int(self.config.semantic_dim)

    def forward(self, visual_feature: Tensor) -> dict[str, Tensor]:
        """Return the unchanged instance vector and normalized semantic vector.

        ``visual_feature`` may have any leading dimensions, followed by
        ``input_dim``.  The instance branch intentionally aliases the floating
        input tensor instead of applying normalization or a learned layer.
        """

        x = torch.as_tensor(visual_feature)
        if x.ndim < 1 or x.shape[-1] != self.input_dim:
            raise ValueError(f"expected [..., {self.input_dim}], got {tuple(x.shape)}")
        x = x.float()
        semantic = self.projection(x)
        if self.residual is not None:
            semantic = semantic + self.residual(x)
        return {"z_instance": x, "z_semantic": F.normalize(semantic, dim=-1)}

    def metadata(self) -> dict[str, Any]:
        return {
            "architecture": "LayerNorm -> Linear -> GELU -> Linear with optional input residual -> L2",
            "config": asdict(self.config),
            "z_instance": "unchanged frozen visual track feature",
            "z_semantic": "learned category-correspondence projection",
            "forbidden_forward_inputs": [
                "category_id",
                "video_id",
                "physical_track_id",
                "semantic_id",
                "category_text",
                "future_observation",
                "future_track",
                "gt_bbox",
                "StateMemory",
                "controller_action",
            ],
            "train_only_supervision": [
                "same_category_cross_track_positive",
                "different_category_hard_negative",
                "causal_prefix_pair",
            ],
            "test_semantic_accessed": False,
        }


def _validate_batch(embeddings: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    z = F.normalize(torch.as_tensor(embeddings).float(), dim=-1)
    y = torch.as_tensor(labels, device=z.device).long().reshape(-1)
    if z.ndim != 2:
        raise ValueError(f"expected [batch, dim] embeddings, got {tuple(z.shape)}")
    if len(y) != z.shape[0]:
        raise ValueError("labels and embeddings must have the same batch size")
    return z, y


def multi_positive_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
    *,
    group_ids: Tensor | None = None,
    temperature: float = 0.07,
) -> Tensor:
    """SupCon-style multi-positive loss for TRAIN categories only.

    ``group_ids`` can identify the same physical track or source instance;
    those rows are not treated as cross-track positives.  Anchors without a
    valid positive are excluded from the mean rather than receiving a fake
    target.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z, y = _validate_batch(embeddings, labels)
    n = z.shape[0]
    if n < 2:
        return z.sum() * 0.0
    logits = (z @ z.T) / float(temperature)
    non_self = ~torch.eye(n, dtype=torch.bool, device=z.device)
    positive = y[:, None].eq(y[None, :]) & non_self
    if group_ids is not None:
        groups = torch.as_tensor(group_ids, device=z.device).reshape(-1)
        if len(groups) != n:
            raise ValueError("group_ids and embeddings must have the same batch size")
        positive &= ~groups[:, None].eq(groups[None, :])
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return z.sum() * 0.0
    logits = logits.masked_fill(~non_self, torch.finfo(logits.dtype).min)
    positive_logits = logits.masked_fill(~positive, torch.finfo(logits.dtype).min)
    return (-(torch.logsumexp(positive_logits[valid], dim=1) - torch.logsumexp(logits[valid], dim=1))).mean()


def hard_negative_ranking_loss(
    embeddings: Tensor,
    labels: Tensor,
    *,
    group_ids: Tensor | None = None,
    margin: float = 0.10,
) -> Tensor:
    """Make the hardest different-category example trail a positive pair."""

    if margin < 0:
        raise ValueError("margin must be non-negative")
    z, y = _validate_batch(embeddings, labels)
    n = z.shape[0]
    if n < 3:
        return z.sum() * 0.0
    similarity = z @ z.T
    non_self = ~torch.eye(n, dtype=torch.bool, device=z.device)
    positive = y[:, None].eq(y[None, :]) & non_self
    negative = ~y[:, None].eq(y[None, :]) & non_self
    if group_ids is not None:
        groups = torch.as_tensor(group_ids, device=z.device).reshape(-1)
        if len(groups) != n:
            raise ValueError("group_ids and embeddings must have the same batch size")
        positive &= ~groups[:, None].eq(groups[None, :])
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not bool(valid.any()):
        return z.sum() * 0.0
    neg_floor = torch.finfo(similarity.dtype).min
    pos_score = similarity.masked_fill(~positive, neg_floor).max(dim=1).values
    neg_score = similarity.masked_fill(~negative, neg_floor).max(dim=1).values
    return F.relu(float(margin) + neg_score[valid] - pos_score[valid]).mean()


def prefix_consistency_loss(
    semantic_prefix: Tensor,
    semantic_reference: Tensor,
    *,
    valid: Tensor | None = None,
) -> Tensor:
    """Align two causal semantic prefixes without introducing future data."""

    first = F.normalize(torch.as_tensor(semantic_prefix).float(), dim=-1)
    second = F.normalize(torch.as_tensor(semantic_reference).float(), dim=-1)
    if first.shape != second.shape:
        raise ValueError("prefix tensors must have identical shapes")
    values = 1.0 - (first * second).sum(dim=-1)
    if valid is not None:
        mask = torch.as_tensor(valid, device=values.device).bool()
        if mask.shape != values.shape:
            raise ValueError("valid mask must match prefix tensor leading shape")
        if not bool(mask.any()):
            return values.sum() * 0.0
        values = values[mask]
    return values.mean()


__all__ = [
    "SemanticAdapter",
    "SemanticAdapterConfig",
    "hard_negative_ranking_loss",
    "multi_positive_contrastive_loss",
    "prefix_consistency_loss",
]
