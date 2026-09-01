"""Small, causal TrackOCD-native end-to-end semantic graph.

The physical proposal/association tensors are supplied by the frozen Phase26
stream in this phase.  The module keeps those physical rows immutable while
jointly exposing proposal, representation, correspondence and semantic-state
interfaces for TRAIN curriculum experiments.  No labels or IDs are consumed by
the forward path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class CausalTrackEncoder(nn.Module):
    """Causal temporal pooling with a 768-D raw-preserving output."""

    def __init__(self, dim: int = 768, hidden: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.temporal = nn.GRU(dim, hidden, batch_first=True)
        self.residual = nn.Linear(hidden, dim)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # sequence is ordered oldest-to-newest; callers must truncate before call.
        x = self.norm(sequence.float())
        h, _ = self.temporal(x)
        if mask is None:
            last = h[:, -1]
            raw = sequence[:, -1]
        else:
            lengths = mask.long().sum(-1).clamp_min(1)
            last = h[torch.arange(h.shape[0], device=h.device), lengths - 1]
            raw = sequence[torch.arange(sequence.shape[0], device=sequence.device), lengths - 1]
        # The zero initial residual gives a documented raw anchor and allows
        # gradients to learn only an evidence update.
        return F.normalize(raw + 0.10 * torch.tanh(self.residual(last)), dim=-1)


class CausalSupportMemory(nn.Module):
    """Permutation-invariant, causal support evidence and bounded residual."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Tanh())
        self.mix = nn.Linear(dim * 2, dim)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, query: torch.Tensor, support: Optional[torch.Tensor], valid: Optional[torch.Tensor] = None):
        if support is None or support.numel() == 0:
            return query, torch.zeros(query.shape[0], device=query.device), torch.zeros_like(query)
        s = F.normalize(self.proj(support.float()), dim=-1)
        if valid is None:
            valid = torch.ones(s.shape[:-1], device=s.device, dtype=torch.bool)
        vm = valid.bool().unsqueeze(-1)
        count = vm.float().sum(1).clamp_min(1.0)
        ctx = (s * vm.float()).sum(1) / count
        quality = (count.squeeze(-1) / max(float(s.shape[1]), 1)).clamp(0, 1)
        z = F.normalize(query + 0.10 * torch.tanh(self.mix(torch.cat([query, ctx], -1))), dim=-1)
        return z, quality, ctx


class SemanticStateController(nn.Module):
    """Causal evidence accumulator; Defer is explicit and no IDs are used."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.readout = nn.Sequential(nn.LayerNorm(dim + 4), nn.Linear(dim + 4, 64), nn.Tanh(), nn.Linear(64, 3))

    def forward(self, representation: torch.Tensor, state: Optional[torch.Tensor] = None):
        if state is None:
            state = torch.zeros(representation.shape[0], 4, device=representation.device)
        # State channels: evidence, persistence, uncertainty, contradiction.
        score = representation.mean(-1, keepdim=True)
        evidence = 0.95 * state[:, :1] + 0.05 * score
        persistence = torch.clamp(state[:, 1:2] + 1.0, 0, 16)
        uncertainty = 0.95 * state[:, 2:3] + 0.05 * (1.0 - representation.abs().mean(-1, keepdim=True))
        contradiction = 0.95 * state[:, 3:4]
        new_state = torch.cat([evidence, persistence, uncertainty, contradiction], -1)
        logits = self.readout(torch.cat([representation, new_state], -1))
        # Action indices are Defer=0, Commit=1, Reject/unknown=2.  Training
        # and evaluation may apply the registered causal utility; no threshold
        # or StateMemory from an older phase is silently reused.
        return logits, new_state


class EndToEndTrackOCD(nn.Module):
    """Single graph exposing the MOT/OCD interfaces without ID leakage."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.track_encoder = CausalTrackEncoder(dim)
        self.support_memory = CausalSupportMemory(dim)
        self.controller = SemanticStateController(dim)
        self.proposal_quality = nn.Linear(dim + 4, 1)
        self.association = nn.Linear(dim + 4, 1)

    def forward(self, query_sequence: torch.Tensor, support: Optional[torch.Tensor] = None,
                support_valid: Optional[torch.Tensor] = None, state: Optional[torch.Tensor] = None,
                geometry: Optional[torch.Tensor] = None):
        qmask = torch.ones(query_sequence.shape[:2], device=query_sequence.device, dtype=torch.bool)
        raw = F.normalize(query_sequence[:, -1].float(), dim=-1)
        track = self.track_encoder(query_sequence, qmask)
        semantic, support_quality, support_ctx = self.support_memory(track, support, support_valid)
        # Exact raw fallback is part of the public interface: no support or an
        # all-invalid support mask cannot alter the semantic row vector.
        if support is None or support.numel() == 0:
            has_support = torch.zeros(query_sequence.shape[0], dtype=torch.bool, device=query_sequence.device)
        elif support_valid is None:
            has_support = torch.ones(query_sequence.shape[0], dtype=torch.bool, device=query_sequence.device)
        else:
            has_support = support_valid.bool().any(1)
        semantic = torch.where(has_support.unsqueeze(-1), semantic, raw)
        logits, new_state = self.controller(semantic, state)
        if geometry is None:
            geometry = torch.zeros(query_sequence.shape[0], 4, device=query_sequence.device)
        aux = torch.cat([semantic.detach(), geometry.float()], -1)
        return {
            "raw": F.normalize(raw.float(), dim=-1),
            "track": track,
            "semantic": semantic,
            "support_context": support_ctx,
            "support_quality": support_quality,
            "action_logits": logits,
            "state": new_state,
            "proposal_quality": self.proposal_quality(aux).squeeze(-1),
            "association": self.association(aux).squeeze(-1),
        }


@dataclass(frozen=True)
class ContractInfo:
    output_dim: int = 768
    invalid_support_exact_raw: bool = True
    physical_ids_mutable: bool = False
    causal: bool = True
