"""Causal feature materialisation and deterministic stream order."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from .data import PREFIXES
from .pair_cache import pair_id


def bank_features(bank: Any, table, cache: dict[str, Any], device: torch.device) -> dict[int, list[dict[str, torch.Tensor]]]:
    out: dict[int, list[dict[str, torch.Tensor]]] = {}
    for prefix in PREFIXES:
        qf = table.get_frame_sequence(bank.query_key, prefix)
        rows: list[dict[str, torch.Tensor]] = []
        for candidate in bank.candidates:
            entry = cache[pair_id(bank.query_key, candidate, prefix, 16)]
            cf = table.get_frame_sequence(candidate, 16)
            qi = np.asarray(entry.get("q_indices", []), dtype=np.int64)
            ci = np.asarray(entry.get("c_indices", []), dtype=np.int64)
            q_t = torch.as_tensor(qf, dtype=torch.float32, device=device)
            c_t = torch.as_tensor(cf, dtype=torch.float32, device=device)
            if len(qi):
                q_m = q_t[torch.as_tensor(qi, dtype=torch.long, device=device)]
                c_m = c_t[torch.as_tensor(ci, dtype=torch.long, device=device)]
                token = torch.cat([torch.abs(q_m - c_m), q_m * c_m], dim=-1)
            else:
                token = torch.zeros((0, 1536), dtype=torch.float32, device=device)
            quality = torch.as_tensor(entry.get("quality_features", []), dtype=torch.float32, device=device).reshape(-1, 5)
            rows.append({
                "pair_tokens": token,
                "quality_features": quality,
                "summary": torch.as_tensor(entry["summary"], dtype=torch.float32, device=device),
                "raw": torch.as_tensor(float(entry["raw_cosine"]), dtype=torch.float32, device=device),
            })
        out[prefix] = rows
    return out


def score_bank(model, features: dict[int, list[dict[str, torch.Tensor]]], prefix: int, *, raw_scores: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if raw_scores is None:
        raw_scores = torch.stack([x["raw"] for x in features[prefix]])
    finals: list[torch.Tensor] = []; deltas: list[torch.Tensor] = []; gates: list[torch.Tensor] = []; bank_gates: list[torch.Tensor] = []; bank_logits: list[torch.Tensor] = []
    for idx, item in enumerate(features[prefix]):
        out = model(item["pair_tokens"], item["quality_features"], item["summary"], item["raw"], raw_scores)
        finals.append(out["final"].reshape(())); deltas.append(out["delta_bounded"].reshape(())); gates.append(out["gate"].reshape(())); bank_gates.append(out["bank_gate"].reshape(())); bank_logits.append(out["bank_gate_logit"].reshape(()))
    return {"final": torch.stack(finals), "raw": raw_scores, "delta_bounded": torch.stack(deltas), "gate": torch.stack(gates), "bank_gate": torch.stack(bank_gates), "bank_gate_logit": torch.stack(bank_logits)}


def deterministic_order(items: list[Any], seed: int) -> list[int]:
    import hashlib
    return sorted(range(len(items)), key=lambda i: (hashlib.sha256(f"{seed}:{items[i].episode_id}:{i}".encode()).hexdigest(), i))


class BankFeatureLRU:
    def __init__(self, table, pair_cache: dict[str, Any], device: torch.device, capacity: int = 8):
        self.table = table; self.pair_cache = pair_cache; self.device = device; self.capacity = capacity; self.data: OrderedDict[int, Any] = OrderedDict()

    def get(self, index: int, bank: Any):
        if index in self.data:
            value = self.data.pop(index); self.data[index] = value; return value
        value = bank_features(bank, self.table, self.pair_cache, self.device)
        self.data[index] = value
        while len(self.data) > self.capacity:
            self.data.popitem(last=False)
        return value
