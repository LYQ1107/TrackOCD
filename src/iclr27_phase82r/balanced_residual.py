"""Balanced two-stage causal residual for the Phase82R route.

The module consumes only the serialized causal observations.  Stage A predicts
whether a reconnect should be attempted; Stage B ranks the candidate fragment
only when a reconnect exists.  Track/category labels never enter the tensors.
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


class BalancedResidualGate(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN, max_candidates: int = MAX_CANDIDATES):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.max_candidates = max_candidates
        self.obs_proj = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, hidden), nn.GELU())
        self.temporal = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.current_proj = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, hidden), nn.GELU())
        self.pair_head = nn.Sequential(nn.LayerNorm(hidden * 4 + 8), nn.Linear(hidden * 4 + 8, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # The gate starts at a neutral p=.5; class-balanced batches, rather
        # than a positive prior, determine the learned policy.
        # current embedding plus top1/top2/margin/count/entropy (five scalars)
        self.gate_head = nn.Sequential(nn.LayerNorm(hidden + 5), nn.Linear(hidden + 5, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    @staticmethod
    def _last(encoded: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        n, k, _ = encoded.shape
        has = valid.bool().any(dim=1)
        idx = valid.bool().long().sum(dim=1).clamp(min=1) - 1
        out = encoded[torch.arange(n, device=encoded.device), idx]
        return out * has.to(out.dtype).unsqueeze(-1)

    def forward(self, current: torch.Tensor, history: torch.Tensor, candidate_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if current.ndim != 2 or history.ndim != 4:
            raise ValueError("current [B,D], history [B,M,K,D] expected")
        b, m, k, d = history.shape
        if d != self.obs_dim or current.shape != (b, d) or m != self.max_candidates:
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
        geometry = history[:, :, -1, :8] - current[:, None, :8]
        pair = torch.cat((cur_expand, traj, torch.abs(cur_expand - traj), cur_expand * traj, geometry), dim=-1)
        candidate_logits = self.pair_head(pair).squeeze(-1).masked_fill(~candidate_mask, -1e4)
        count = candidate_mask.float().sum(dim=1, keepdim=True) / float(max(1, m))
        top = candidate_logits.topk(k=2, dim=1).values if m >= 2 else torch.cat((candidate_logits, candidate_logits), dim=1).topk(k=2, dim=1).values
        top1, top2 = top[:, 0:1], top[:, 1:2]
        margin = top1 - top2
        probs = candidate_logits.masked_fill(~candidate_mask, -1e4).softmax(dim=1)
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / 3.0
        gate_features = torch.cat((cur, top1, top2, margin, count, entropy), dim=1)
        gate_logit = self.gate_head(gate_features).squeeze(-1)
        return {"candidate_logits": candidate_logits, "gate_logit": gate_logit, "gate_features": gate_features}


def balanced_loss(output: dict[str, torch.Tensor], target: torch.Tensor, candidate_mask: torch.Tensor, rank_weight: float = 1.0, safety_weight: float = 0.25) -> tuple[torch.Tensor, dict[str, float]]:
    """Train gate with balanced BCE and candidate CE on positive examples.

    ``target`` is 0 for KEEP_Q0 and 1..M for the registered reconnect
    candidate.  Safety penalty is applied only to negative examples, preserving
    the exact raw fallback for p<.5 at inference.
    """
    gate_target = (target > 0).float()
    bce = F.binary_cross_entropy_with_logits(output["gate_logit"], gate_target)
    pos = target > 0
    rank = output["candidate_logits"].new_zeros(())
    if pos.any():
        rank = F.cross_entropy(output["candidate_logits"][pos].masked_fill(~candidate_mask[pos], -1e4), (target[pos] - 1).long())
    p = torch.sigmoid(output["gate_logit"])
    safety = (p[~pos] ** 2).mean() if (~pos).any() else p.new_zeros(())
    loss = bce + rank_weight * rank + safety_weight * safety
    with torch.no_grad():
        pred_gate = p >= 0.5
        candidate_pred = output["candidate_logits"].argmax(dim=1) + 1
        pred = torch.where(pred_gate, candidate_pred, torch.zeros_like(candidate_pred))
        metrics = {
            "loss": float(loss.detach().cpu()), "bce": float(bce.detach().cpu()), "rank": float(rank.detach().cpu()), "safety": float(safety.detach().cpu()),
            "gate_use_rate": float(pred_gate.float().mean().cpu()), "pred_reconnect_rate": float((pred > 0).float().mean().cpu()),
            "target_reconnect_rate": float(pos.float().mean().cpu()), "accuracy": float((pred == target).float().mean().cpu()),
        }
    return loss, metrics


def predict(output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    p = torch.sigmoid(output["gate_logit"])
    candidate = output["candidate_logits"].argmax(dim=1) + 1
    chosen = torch.where(p >= 0.5, candidate, torch.zeros_like(candidate))
    return chosen, p


def contract_summary(model: nn.Module) -> dict[str, Any]:
    return {
        "model": "BalancedResidualGate", "obs_dim": OBS_DIM, "history_length": HISTORY_LENGTH,
        "hidden": HIDDEN, "temporal": "single_layer_GRU", "max_candidates": MAX_CANDIDATES,
        "parameters": sum(p.numel() for p in model.parameters()),
        "stage_a": "class-balanced BCE gate p>=0.5", "stage_b": "positive-only candidate CE ranking",
        "inference_forbidden": ["category_id", "track_id", "physical_id", "semantic_id", "future", "held_gt", "text"],
    }
