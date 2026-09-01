"""Deferred Semantic Tracklet Memory (DSTM)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class DSTM(nn.Module):
    """Compact causal sequence encoder and set-conditioned state decoder.

    Physical track IDs are deliberately absent from every tensor accepted by
    this module. Candidate positions are permuted by the episode builder.
    """

    def __init__(self, input_dim: int, hidden_dim: int, projection_dim: int,
                 known_count: int, max_training_states: int = 6,
                 no_history: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.known_count = known_count
        self.max_training_states = max_training_states
        self.no_history = no_history
        self.row_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim), nn.LayerNorm(projection_dim), nn.GELU(),
            nn.Linear(projection_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.tracklet_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.state_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim), nn.LayerNorm(projection_dim), nn.GELU(),
            nn.Linear(projection_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1)
        )
        self.known_aux = nn.Linear(hidden_dim, known_count)
        self.known_tokens = nn.Parameter(torch.empty(known_count, hidden_dim))
        self.new_token = nn.Parameter(torch.empty(1, hidden_dim))
        self.defer_token = nn.Parameter(torch.empty(1, hidden_dim))
        self.type_embedding = nn.Parameter(torch.empty(4, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=.10, activation="gelu", batch_first=True, norm_first=True,
        )
        self.state_set_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.query_cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=.10, batch_first=True
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_bias = nn.Parameter(torch.zeros(4))
        self._reset()

    def _reset(self) -> None:
        nn.init.normal_(self.known_tokens, std=.02)
        nn.init.normal_(self.new_token, std=.02)
        nn.init.normal_(self.defer_token, std=.02)
        nn.init.normal_(self.type_embedding, std=.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_sequence(self, rows: torch.Tensor, lengths: torch.Tensor,
                        return_all: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode padded causal prefixes; only positions <= lengths are read."""
        if self.no_history:
            gather = (lengths - 1).clamp_min(0)[:, None, None].expand(-1, 1, rows.shape[-1])
            rows = rows.gather(1, gather)
            lengths = torch.ones_like(lengths)
        projected = self.row_projection(rows)
        output, _ = self.tracklet_gru(projected)
        gather = (lengths - 1).clamp_min(0)[:, None, None].expand(-1, 1, self.hidden_dim)
        token = output.gather(1, gather).squeeze(1)
        token = F.normalize(token, dim=-1)
        return token, output if return_all else None

    def encode_state(self, state_rows: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.state_projection(state_rows), dim=-1)

    def decode(self, query: torch.Tensor, state_tokens: torch.Tensor,
               state_mask: torch.Tensor, known_mask: torch.Tensor | None = None,
               allow_defer: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode over known, variable semantic states, NEW and DEFER."""
        batch, state_count, _ = state_tokens.shape
        known = self.known_tokens[None].expand(batch, -1, -1)
        new = self.new_token[None].expand(batch, -1, -1)
        defer = self.defer_token[None].expand(batch, -1, -1)
        candidates = torch.cat([known, state_tokens, new, defer], dim=1)
        type_id = torch.cat([
            torch.zeros(self.known_count, dtype=torch.long, device=query.device),
            torch.ones(state_count, dtype=torch.long, device=query.device),
            torch.tensor([2, 3], dtype=torch.long, device=query.device),
        ])
        candidates = candidates + self.type_embedding[type_id][None]
        if known_mask is None:
            known_mask = torch.ones(batch, self.known_count, dtype=torch.bool, device=query.device)
        special = torch.ones(batch, 2, dtype=torch.bool, device=query.device)
        valid = torch.cat([known_mask, state_mask, special], dim=1)
        if not allow_defer:
            valid[:, -1] = False
        contextual = self.state_set_encoder(candidates, src_key_padding_mask=~valid)
        attended, weights = self.query_cross_attention(
            query[:, None], contextual, contextual, key_padding_mask=~valid,
            need_weights=True,
        )
        conditioned = self.query_norm(query + attended[:, 0])
        q = self.query_projection(conditioned)
        k = self.candidate_projection(contextual)
        logits = torch.einsum("bd,bnd->bn", q, k) / math.sqrt(self.hidden_dim)
        logits = logits + self.candidate_bias[type_id][None]
        logits = logits.masked_fill(~valid, -1e4)
        return logits, {"attention": weights[:, 0], "conditioned_query": conditioned,
                        "contextual_candidates": contextual, "valid_candidates": valid}

    def forward(self, query_rows: torch.Tensor, query_lengths: torch.Tensor,
                state_rows: torch.Tensor, state_mask: torch.Tensor,
                known_mask: torch.Tensor | None = None,
                allow_defer: bool = True, return_all: bool = False) -> dict[str, Any]:
        query, sequence = self.encode_sequence(query_rows, query_lengths, return_all=return_all)
        states = self.encode_state(state_rows)
        logits, aux = self.decode(query, states, state_mask, known_mask, allow_defer)
        return {
            "query": query, "sequence": sequence, "state_tokens": states,
            "logits": logits, "reliability_logit": self.reliability_head(query).squeeze(-1),
            "known_aux_logits": self.known_aux(query), **aux,
        }


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
