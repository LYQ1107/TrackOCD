"""Phase82P Q0-anchored residual fragment repair model.

Only causal observation vectors are consumed.  Candidate/track IDs are used by
the caller for bookkeeping and are never concatenated to a tensor.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

OBS_DIM = 49
HISTORY_LENGTH = 8
MAX_CANDIDATES = 16
HIDDEN = 256


class ResidualTrajectoryEncoder(nn.Module):
    """Two-layer temporal Transformer over each candidate's K observations."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN, layers: int = 2, heads: int = 4, max_candidates: int = MAX_CANDIDATES):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.max_candidates = max_candidates
        self.history_proj = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, hidden), nn.GELU())
        self.current_proj = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, hidden), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden * 2, dropout=0.1, batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.pair_head = nn.Sequential(nn.LayerNorm(hidden * 4 + 8), nn.Linear(hidden * 4 + 8, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.keep_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        # A positive KEEP prior is an architectural safety anchor, not a tuned
        # inference threshold; valid evidence must overcome it during training.
        nn.init.constant_(self.keep_head[-1].bias, 0.75)

    @staticmethod
    def _pool(encoded: torch.Tensor, valid_steps: torch.Tensor) -> torch.Tensor:
        # ``valid_steps`` is [N,K].  Ensure an all-padding history still has a
        # finite zero token for Transformer key-padding semantics.
        valid = valid_steps.bool()
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone(); valid[empty, 0] = True
        positions = torch.arange(encoded.shape[1], device=encoded.device).view(1, -1).expand(encoded.shape[0], -1)
        last = (positions * valid.long()).max(dim=1).values
        pooled = encoded[torch.arange(encoded.shape[0], device=encoded.device), last]
        pooled = pooled * (~empty).to(pooled.dtype).unsqueeze(-1)
        return pooled

    def forward(self, current: torch.Tensor, history: torch.Tensor, candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return [B, 1+M] logits (column 0 is KEEP_Q0)."""
        if current.ndim != 2 or history.ndim != 4:
            raise ValueError("current [B,D], history [B,M,K,D] expected")
        b, m, k, d = history.shape
        if d != self.obs_dim or current.shape != (b, d):
            raise ValueError(f"shape mismatch current={tuple(current.shape)} history={tuple(history.shape)}")
        if candidate_mask is None:
            candidate_mask = history.abs().sum(dim=(-1, -2)) > 1e-8
        candidate_mask = candidate_mask.bool()
        hist_valid = history.abs().sum(dim=-1) > 1e-8
        flat = history.reshape(b * m, k, d)
        encoded = self.history_proj(flat)
        empty = ~hist_valid.reshape(b * m, k).any(dim=1)
        key_padding = ~hist_valid.reshape(b * m, k)
        if empty.any():
            key_padding = key_padding.clone(); key_padding[empty, 0] = False
        encoded = self.temporal(encoded, src_key_padding_mask=key_padding)
        traj = self._pool(encoded, hist_valid.reshape(b * m, k)).reshape(b, m, self.hidden)
        cur = self.current_proj(current)
        cur_expand = cur.unsqueeze(1).expand(-1, m, -1)
        pair_geometry = history[:, :, -1, :8] - current[:, None, :8]
        pair = torch.cat((cur_expand, traj, cur_expand - traj, cur_expand * traj, pair_geometry), dim=-1)
        candidate_logits = self.pair_head(pair).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask, -1e4)
        keep = self.keep_head(cur).squeeze(-1)
        return torch.cat((keep.unsqueeze(-1), candidate_logits), dim=-1)


def residual_loss(logits: torch.Tensor, target: torch.Tensor, candidate_mask: torch.Tensor, false_reconnect_weight: float = 2.0, missed_repair_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Listwise CE with a fixed asymmetric KEEP safety weight.

    Target 0 means KEEP_Q0 and is weighted as the false-reconnect-safe class;
    positive targets are the registered missed-repair supervision.  No GT field
    is read by this function.
    """
    valid = torch.cat((torch.ones((candidate_mask.shape[0], 1), dtype=torch.bool, device=logits.device), candidate_mask.bool()), dim=1)
    safe_logits = logits.masked_fill(~valid, -1e4)
    weights = torch.where(target == 0, torch.as_tensor(false_reconnect_weight, device=logits.device), torch.as_tensor(missed_repair_weight, device=logits.device))
    per = F.cross_entropy(safe_logits, target.long(), reduction="none")
    loss = (per * weights).mean()
    with torch.no_grad():
        pred = safe_logits.argmax(dim=1)
        metrics = {
            "loss": float(loss.detach().cpu()),
            "accuracy": float((pred == target).float().mean().cpu()),
            "keep_rate": float((pred == 0).float().mean().cpu()),
            "target_reconnect_rate": float((target > 0).float().mean().cpu()),
            "pred_reconnect_rate": float((pred > 0).float().mean().cpu()),
            "false_reconnect_rate": float(((target == 0) & (pred > 0)).float().mean().cpu()),
            "correct_reconnect_rate": float(((target > 0) & (pred == target)).float().mean().cpu()),
        }
    return loss, metrics


def contract_summary(model: nn.Module) -> dict[str, Any]:
    params = sum(p.numel() for p in model.parameters())
    return {"model": "ResidualTrajectoryEncoder", "obs_dim": OBS_DIM, "history_length": HISTORY_LENGTH, "hidden": HIDDEN, "temporal_layers": 2, "temporal_heads": 4, "max_candidates": MAX_CANDIDATES, "parameters": params, "inference_forbidden": ["category_id", "track_id", "physical_id", "semantic_id", "future", "held_gt", "text"]}
