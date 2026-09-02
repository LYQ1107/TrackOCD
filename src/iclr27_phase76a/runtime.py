"""Runtime helpers shared by training and TRAIN-disjoint validation."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from .candidate_bank import CandidateBank
from .pair_cache import pair_id
from .raw_anchor import raw_mean_cosine


def bank_features(bank: CandidateBank, table, cache: dict[str, Any], device: torch.device) -> dict[int, list[dict[str, torch.Tensor]]]:
    """Materialise pair tokens from frozen table plus detached index cache."""
    out: dict[int, list[dict[str, torch.Tensor]]] = {}
    for prefix in (1, 2, 4, 8, 16):
        candidates = []
        qf = table.get_frame_sequence(bank.query_key, prefix)
        for c, raw in zip(bank.candidates, bank.raw_scores):
            entry = cache[pair_id(bank.query_key, c, prefix, 16)]
            cf = table.get_frame_sequence(c, 16)
            qi = np.asarray(entry["q_indices"], dtype=np.int64); ci = np.asarray(entry["c_indices"], dtype=np.int64)
            q_t = torch.as_tensor(qf, dtype=torch.float32, device=device); c_t = torch.as_tensor(cf, dtype=torch.float32, device=device)
            if len(qi):
                q_m, c_m = q_t[torch.as_tensor(qi, device=device)], c_t[torch.as_tensor(ci, device=device)]
                pair = torch.cat([torch.abs(q_m - c_m), q_m * c_m], dim=-1)
            else:
                pair = torch.zeros((0, 1536), dtype=torch.float32, device=device)
            # The raw anchor is prefix-specific.  ``bank.raw_scores`` is a
            # p16 metadata convenience only; use the detached cache value for
            # every causal prefix.
            candidates.append({"pair_tokens": pair, "summary": torch.as_tensor(entry["summary"], dtype=torch.float32, device=device), "raw": torch.tensor(float(entry["raw_cosine"]), dtype=torch.float32, device=device)})
        out[prefix] = candidates
    return out


def score_bank(model, features: dict[int, list[dict[str, torch.Tensor]]], prefix: int) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    finals: list[torch.Tensor] = []; deltas: list[torch.Tensor] = []; confidences: list[torch.Tensor] = []
    for item in features[prefix]:
        y = model(item["pair_tokens"], item["summary"], item["raw"])
        finals.append(y["final"].reshape(())); deltas.append(y["delta"].reshape(())); confidences.append(y["confidence"].reshape(()))
    return torch.stack(finals), deltas, confidences


def deterministic_order(banks: list[CandidateBank], seed: int) -> list[int]:
    # Hash sorting, rather than a per-update RNG, makes episode visits exactly
    # reproducible after checkpoint resume.
    import hashlib
    return sorted(range(len(banks)), key=lambda i: (hashlib.sha256(f"{seed}:{banks[i].episode_id}:{i}".encode()).hexdigest(), i))


class BankFeatureLRU:
    def __init__(self, table, pair_cache: dict[str, Any], device: torch.device, capacity: int = 16):
        self.table = table; self.pair_cache = pair_cache; self.device = device; self.capacity = capacity; self.data: OrderedDict[int, Any] = OrderedDict()

    def get(self, index: int, bank: CandidateBank):
        if index in self.data:
            value = self.data.pop(index); self.data[index] = value; return value
        value = bank_features(bank, self.table, self.pair_cache, self.device)
        self.data[index] = value
        while len(self.data) > self.capacity:
            self.data.popitem(last=False)
        return value


def raw_scores_for_bank(bank: CandidateBank) -> np.ndarray:
    return np.asarray(bank.raw_scores, dtype=np.float32)
