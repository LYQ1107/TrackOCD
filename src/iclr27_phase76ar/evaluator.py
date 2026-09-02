"""Exact TRAIN-disjoint retrieval evaluator for Phase76AR streams."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch

from src.iclr27_phase75d.retrieval_metrics import score_records

from .data import PREFIXES
from .losses import teacher_use
from .runtime import BankFeatureLRU, score_bank


def _indices(bank: Any) -> tuple[list[int], list[int]]:
    candidates = list(bank.candidates)
    positives = set(bank.positives if hasattr(bank, "positives") else bank.positive_keys)
    negatives = set(bank.negatives if hasattr(bank, "negatives") else bank.negative_keys)
    return [i for i, key in enumerate(candidates) if key in positives], [i for i, key in enumerate(candidates) if key in negatives]


def evaluate_banks(model, banks: list[Any], table, pair_cache: dict[str, Any], device: torch.device, *, limit: int | None = None, indices: list[int] | None = None) -> dict[str, Any]:
    selected_indices = list(range(len(banks))) if indices is None else list(indices)
    if limit is not None:
        selected_indices = selected_indices[: int(limit)]
    selected = [banks[i] for i in selected_indices]
    lru = BankFeatureLRU(table, pair_cache, device, capacity=8)
    prefix_rows: list[dict[str, Any]] = []
    for prefix in PREFIXES:
        records: list[dict[str, Any]] = []
        teacher_rows: list[dict[str, Any]] = []
        for local_idx, bank in enumerate(selected):
            feat = lru.get(selected_indices[local_idx], bank)
            raw_scores = torch.stack([x["raw"] for x in feat[prefix]])
            scored = score_bank(model, feat, prefix, raw_scores=raw_scores)
            scores = scored["final"].detach().cpu().numpy().astype(np.float32).tolist()
            raw = scored["raw"].detach().cpu().numpy().astype(np.float32).tolist()
            pos_idx, neg_idx = _indices(bank)
            teacher = teacher_use(raw_scores, pos_idx, feat[prefix], neg_idx)
            gate = float(scored["bank_gate"].mean().detach().cpu()) if len(feat[prefix]) else 0.0
            # A relation intervention means that the actual top-1 changes,
            # not merely that a sigmoid is nonzero.
            raw_top = int(np.argmax(raw)) if raw else -1
            learned_top = int(np.argmax(scores)) if scores else -1
            teacher_decision = bool(teacher >= 0.5)
            learned_decision = bool(gate >= 0.5)
            teacher_rows.append({
                "query_key": bank.query_key, "episode_id": bank.episode_id,
                "teacher_use": teacher, "bank_gate": gate,
                "teacher_agreement": int(teacher_decision == learned_decision),
                "raw_top1_positive": int(raw_top in pos_idx), "learned_top1_positive": int(learned_top in pos_idx),
                "intervened": int(learned_top != raw_top),
                "raw_top1": raw_top, "learned_top1": learned_top,
            })
            records.append({
                "query_key": bank.query_key, "category": bank.category, "video": bank.video,
                "candidates": list(bank.candidates), "positives": list(bank.positives if hasattr(bank, "positives") else bank.positive_keys),
                "negatives": list(bank.negatives if hasattr(bank, "negatives") else bank.negative_keys),
                "scores": scores, "raw_scores": raw,
            })
        metric = score_records(records)
        teacher_rate = float(np.mean([r["teacher_use"] for r in teacher_rows])) if teacher_rows else 0.0
        agreement = float(np.mean([r["teacher_agreement"] for r in teacher_rows])) if teacher_rows else 0.0
        intervention = int(sum(r["intervened"] for r in teacher_rows))
        metric.update({
            "scope": "phase76ar_train_disjoint", "teacher_use_rate": teacher_rate,
            "teacher_agreement": agreement, "intervention_count": intervention,
            "intervention_rate": float(intervention / max(len(teacher_rows), 1)),
            "mean_bank_gate": float(np.mean([r["bank_gate"] for r in teacher_rows])) if teacher_rows else 0.0,
            "teacher_rows": teacher_rows,
        })
        prefix_rows.append({"prefix": prefix, "learned": metric, "raw": {k: metric.get(f"raw_{k}") for k in ("r1", "r5", "map", "hard_negative_gap")}, "queries": len(records), "candidate_count": int(sum(len(r["candidates"]) for r in records))})
    return {"prefix_rows": prefix_rows, "queries": len(selected), "selected_indices": selected_indices}


def p16(result: dict[str, Any]) -> dict[str, Any]:
    metric = dict(next(x for x in result["prefix_rows"] if x["prefix"] == 16)["learned"])
    for key in ("r1", "map", "hard_negative_gap"):
        metric[f"delta_{key if key != 'hard_negative_gap' else 'hard_gap'}"] = float(metric[key] - metric[f"raw_{key}"])
    return metric
