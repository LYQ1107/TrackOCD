"""Train-disjoint validation for raw anchor and local relation reranker."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.iclr27_phase75d.retrieval_metrics import score_records

from .candidate_bank import CandidateBank
from .runtime import BankFeatureLRU, score_bank


def evaluate_banks(model, banks: list[CandidateBank], table, pair_cache: dict[str, Any], device, *, limit: int | None = None) -> dict[str, Any]:
    selected = banks if limit is None else banks[: int(limit)]
    lru = BankFeatureLRU(table, pair_cache, device, capacity=8)
    prefix_rows = []
    for prefix in (1, 2, 4, 8, 16):
        records = []
        for i, bank in enumerate(selected):
            feat = lru.get(i, bank)
            learned, _, _ = score_bank(model, feat, prefix)
            scores = learned.detach().cpu().numpy().astype(np.float32).tolist()
            raw_scores = [float(x) for x in (pair_cache[f"{bank.query_key}|16|{c}|{prefix}"]["raw_cosine"] for c in bank.candidates)]
            records.append({"query_key": bank.query_key, "category": bank.category, "video": bank.video, "candidates": list(bank.candidates), "positives": list(bank.positives), "negatives": list(bank.negatives), "scores": scores, "raw_scores": raw_scores})
        metric = score_records(records)
        metric.pop("per_query", None)
        metric["scope"] = "phase30_val_candidate_bank" if limit is None else "bounded_phase30_val_candidate_bank"
        prefix_rows.append({"prefix": prefix, "learned": metric, "raw": {k: metric.get(f"raw_{k}") for k in ("r1", "r5", "map", "hard_negative_gap")}, "candidate_count": int(sum(len(b.candidates) for b in selected)), "queries": metric.get("queries", 0)})
    return {"prefix_rows": prefix_rows, "queries": len(selected), "candidate_banks": len(selected)}

