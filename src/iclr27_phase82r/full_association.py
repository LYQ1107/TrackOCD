"""Trainable, class-agnostic causal association over an augmented candidate set.

The model is deliberately separate from :mod:`balanced_residual`.  It predicts
one explicit NEW action together with a masked score for prior causal tracks.
Only numerical observations (appearance, geometry and history) are consumed;
the physical/semantic labels used to build ``target`` never enter ``forward``.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

OBS_DIM = 48
HISTORY_LENGTH = 8
MAX_CANDIDATES = 16
HIDDEN = 96


class FullAssociation(nn.Module):
    """Single-layer causal scorer with an explicit NEW dummy candidate."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden: int = HIDDEN,
        max_candidates: int = MAX_CANDIDATES,
        explicit_app_cosine: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.max_candidates = max_candidates
        self.explicit_app_cosine = bool(explicit_app_cosine)
        self.obs_proj = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, hidden), nn.GELU())
        self.temporal = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.current_proj = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, hidden), nn.GELU())
        # Four hidden vectors (current, candidate, abs-difference, product) plus
        # eight causal geometric deltas from the parent observation.
        pair_dim = hidden * 4 + 8 + (1 if self.explicit_app_cosine else 0)
        self.pair_head = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.new_head = nn.Sequential(nn.LayerNorm(hidden + 5), nn.Linear(hidden + 5, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    @staticmethod
    def _last(encoded: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        n, k, _ = encoded.shape
        has = valid.bool().any(dim=1)
        # Histories are right-padded, so the last valid position is count-1.
        idx = valid.bool().long().sum(dim=1).clamp(min=1) - 1
        out = encoded[torch.arange(n, device=encoded.device), idx]
        return out * has.to(out.dtype).unsqueeze(-1)

    def forward(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if current.ndim != 2 or history.ndim != 4:
            raise ValueError("current [B,D], history [B,M,K,D] expected")
        b, m, k, d = history.shape
        if d != self.obs_dim or current.shape != (b, d) or m != self.max_candidates or k != HISTORY_LENGTH:
            raise ValueError(f"shape mismatch current={tuple(current.shape)} history={tuple(history.shape)}")
        if candidate_mask is None:
            candidate_mask = history.abs().sum(dim=(-1, -2)) > 1e-8
        candidate_mask = candidate_mask.bool()
        valid = history.abs().sum(dim=-1) > 1e-8
        flat = history.reshape(b * m, k, d)
        enc, _ = self.temporal(self.obs_proj(flat))
        traj = self._last(enc, valid.reshape(b * m, k)).reshape(b, m, self.hidden)
        cur = self.current_proj(current)
        cur_expand = cur.unsqueeze(1).expand(-1, m, -1)
        # The first eight observation values are normalized box/center/size.
        geometry = history[:, :, -1, :8] - current[:, None, :8]
        pieces = [cur_expand, traj, torch.abs(cur_expand - traj), cur_expand * traj, geometry]
        if self.explicit_app_cosine:
            current_app = current[:, None, -32:]
            candidate_app = history[:, :, -1, -32:]
            app_cos = (current_app * candidate_app).sum(dim=-1, keepdim=True) / (current_app.norm(dim=-1, keepdim=True) * candidate_app.norm(dim=-1, keepdim=True)).clamp_min(1e-8)
            pieces.append(app_cos)
        pair = torch.cat(pieces, dim=-1)
        candidate_logits = self.pair_head(pair).squeeze(-1).masked_fill(~candidate_mask, -1e4)
        count = candidate_mask.float().sum(dim=1, keepdim=True) / float(max(1, m))
        top = candidate_logits.topk(k=2, dim=1).values if m >= 2 else torch.cat((candidate_logits, candidate_logits), dim=1).topk(k=2, dim=1).values
        top1, top2 = top[:, 0:1], top[:, 1:2]
        margin = top1 - top2
        probs = candidate_logits.softmax(dim=1)
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / 3.0
        new_features = torch.cat((cur, top1, top2, margin, count, entropy), dim=1)
        new_logit = self.new_head(new_features).squeeze(-1)
        return {"candidate_logits": candidate_logits, "new_logit": new_logit, "new_features": new_features}


def assignment_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    candidate_mask: torch.Tensor,
    safety_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked CE over ``0=NEW`` and ``1..M=existing candidate``."""
    logits = torch.cat((output["new_logit"].unsqueeze(1), output["candidate_logits"]), dim=1)
    # Invalid candidates are already -1e4; explicitly keep the NEW option.
    loss = F.cross_entropy(logits, target.long())
    pred = logits.argmax(dim=1)
    pos = target > 0
    pred_existing = pred > 0
    # A small conservative penalty discourages arbitrary existing assignments
    # on examples whose TRAIN label is NEW; it does not use any label at test.
    safety = ((~pos) & pred_existing).float().mean() * safety_weight
    total = loss + safety
    with torch.no_grad():
        existing_correct = (pos & (pred == target)).sum()
        metrics = {
            "loss": float(total.detach().cpu()),
            "cross_entropy": float(loss.detach().cpu()),
            "safety": float(safety.detach().cpu()),
            "accuracy": float((pred == target).float().mean().cpu()),
            "target_existing_rate": float(pos.float().mean().cpu()),
            "pred_existing_rate": float(pred_existing.float().mean().cpu()),
            "existing_precision": float(existing_correct.cpu() / pred_existing.sum().clamp_min(1)),
            "existing_recall": float(existing_correct.cpu() / pos.sum().clamp_min(1)),
        }
    return total, metrics


def predict(output: dict[str, torch.Tensor], candidate_mask: torch.Tensor) -> torch.Tensor:
    logits = torch.cat((output["new_logit"].unsqueeze(1), output["candidate_logits"]), dim=1)
    pred = logits.argmax(dim=1)
    return torch.where(candidate_mask.any(dim=1), pred, torch.zeros_like(pred))


def contract_summary(model: nn.Module) -> dict[str, Any]:
    return {
        "model": "FullAssociation",
        "obs_dim": OBS_DIM,
        "history_length": HISTORY_LENGTH,
        "hidden": HIDDEN,
        "temporal": "single_layer_GRU",
            "max_candidates": MAX_CANDIDATES,
        "explicit_app_cosine": bool(getattr(model, "explicit_app_cosine", False)),
        "parameters": sum(p.numel() for p in model.parameters()),
        "actions": "0=NEW, 1..M=causal existing track",
        "inference_tensor_fields": ["normalized_box", "score", "causal_velocity", "fixed_DINOv2_projection", "history"],
        "forbidden_inference_fields": ["category_id", "track_id", "physical_id", "semantic_id", "future", "held_gt", "text", "DEV+", "Q1", "public_new", "sealed"],
    }
