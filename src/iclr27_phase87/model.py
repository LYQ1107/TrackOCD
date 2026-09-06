"""Train/inference-shared causal representation and memory relation models."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CausalTrackEncoder(nn.Module):
    def __init__(self, raw_dim: int = 768, geom_dim: int = 15, hidden: int = 256, residual_scale: float = 0.05) -> None:
        super().__init__()
        self.raw_dim = raw_dim
        self.geom_dim = geom_dim
        self.hidden = hidden
        self.residual_scale = residual_scale
        self.gru = nn.GRU(raw_dim + geom_dim, hidden, num_layers=1, batch_first=True)
        self.residual = nn.Sequential(nn.Linear(hidden + raw_dim, 256), nn.GELU(), nn.Linear(256, raw_dim))

    def forward_sequence(self, raw_seq: Tensor, geom_seq: Tensor, mask: Tensor) -> dict[str, Tensor]:
        raw_seq = F.normalize(raw_seq.float(), dim=-1)
        hidden_seq, _ = self.gru(torch.cat([raw_seq, geom_seq.float()], dim=-1))
        latest_index = mask.long().sum(dim=1).clamp_min(1) - 1
        batch = torch.arange(raw_seq.shape[0], device=raw_seq.device)
        latest_raw = raw_seq[batch, latest_index]
        latest_hidden = hidden_seq[batch, latest_index]
        residual = self.residual(torch.cat([hidden_seq.reshape(-1, self.hidden), raw_seq.reshape(-1, self.raw_dim)], dim=-1)).reshape(raw_seq.shape[0], raw_seq.shape[1], self.raw_dim)
        semantic_seq = F.normalize(raw_seq + self.residual_scale * torch.tanh(residual), dim=-1)
        return {"anchor_seq": raw_seq, "semantic_seq": semantic_seq, "track_hidden_seq": hidden_seq, "anchor": latest_raw, "semantic": semantic_seq[batch, latest_index], "track_hidden": latest_hidden}

    def forward(self, raw_seq: Tensor, geom_seq: Tensor, mask: Tensor) -> dict[str, Tensor]:
        return {key: value for key, value in self.forward_sequence(raw_seq, geom_seq, mask).items() if not key.endswith("_seq")}


class MemoryRelationEncoder(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(21), nn.Linear(21, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.repr = nn.Sequential(nn.LayerNorm(21), nn.Linear(21, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, semantic: Tensor, prototypes: Tensor, prototype_mask: Tensor, state_stats: Tensor, previous_evidence: Tensor, support_features: Tensor, quality: Tensor) -> dict[str, Tensor]:
        # semantic [B,D], prototypes [B,S,K,D]
        similarity = torch.einsum("bd,bskd->bsk", F.normalize(semantic, dim=-1), F.normalize(prototypes, dim=-1))
        masked = similarity.masked_fill(~prototype_mask, -1.0)
        valid = prototype_mask.float()
        count = valid.sum(dim=-1).clamp_min(1.0)
        max_sim = masked.max(dim=-1).values
        mean_sim = (masked * valid).sum(dim=-1) / count
        min_sim = torch.where(prototype_mask, similarity, torch.ones_like(similarity)).min(dim=-1).values
        centered = (similarity - mean_sim.unsqueeze(-1)) * valid
        std_sim = torch.sqrt((centered.square().sum(dim=-1) / count).clamp_min(1e-6))
        centroid = (prototypes * valid.unsqueeze(-1)).sum(dim=-2) / count.unsqueeze(-1)
        centroid_sim = (F.normalize(semantic, dim=-1).unsqueeze(1) * F.normalize(centroid, dim=-1)).sum(dim=-1)
        relation = torch.stack([max_sim, mean_sim, min_sim, std_sim, centroid_sim], dim=-1)
        support = support_features.unsqueeze(1).expand(-1, prototypes.shape[1], -1)
        candidate = torch.cat([relation, state_stats, previous_evidence.unsqueeze(-1), quality.view(-1, 1, 1).expand(-1, prototypes.shape[1], -1), support], dim=-1)
        return {"state_logits": self.net(candidate).squeeze(-1), "state_repr": self.repr(candidate)}
