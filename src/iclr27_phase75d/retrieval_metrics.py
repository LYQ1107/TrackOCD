"""Metrics for global and manifest-legal pairwise retrieval."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(r[field]) for r in rows])) if rows else 0.0


def score_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        candidates = list(rec["candidates"])
        positives = set(rec["positives"])
        negatives = set(rec["negatives"])
        if not candidates or not positives or not negatives:
            continue
        scores = np.asarray(rec["scores"], dtype=np.float32)
        raw_scores = np.asarray(rec["raw_scores"], dtype=np.float32)
        if len(scores) != len(candidates) or len(raw_scores) != len(candidates):
            raise ValueError("candidate/score length mismatch")
        order = np.argsort(scores)[::-1]
        raw_order = np.argsort(raw_scores)[::-1]
        hit = np.asarray([int(candidates[int(i)] in positives) for i in order], dtype=np.float32)
        raw_hit = np.asarray([int(candidates[int(i)] in positives) for i in raw_order], dtype=np.float32)
        cum = np.cumsum(hit)
        raw_cum = np.cumsum(raw_hit)
        ap = float(np.sum(cum / (np.arange(len(hit)) + 1) * hit) / max(len(positives), 1))
        raw_ap = float(np.sum(raw_cum / (np.arange(len(raw_hit)) + 1) * raw_hit) / max(len(positives), 1))
        pos_scores = scores[np.asarray([c in positives for c in candidates])]
        neg_scores = scores[np.asarray([c in negatives for c in candidates])]
        raw_pos_scores = raw_scores[np.asarray([c in positives for c in candidates])]
        raw_neg_scores = raw_scores[np.asarray([c in negatives for c in candidates])]
        rows.append({
            "query_key": rec["query_key"], "category": rec.get("category"), "video": rec.get("video"),
            "candidate_count": len(candidates), "positive_count": len(positives), "negative_count": len(negatives),
            "r1": float(hit[0]), "r5": float(hit[:5].max(initial=0.0)), "map": ap,
            "raw_r1": float(raw_hit[0]), "raw_r5": float(raw_hit[:5].max(initial=0.0)), "raw_map": raw_ap,
            "hard_negative_gap": float(np.max(pos_scores) - np.max(neg_scores)),
            "raw_hard_negative_gap": float(np.max(raw_pos_scores) - np.max(raw_neg_scores)),
            "top1": candidates[int(order[0])], "raw_top1": candidates[int(raw_order[0])],
            "unsafe_flip": bool(raw_hit[0] > 0 and hit[0] <= 0),
            "top1_changed": bool(int(order[0]) != int(raw_order[0])),
        })
    by_cat: dict[Any, list[dict[str, Any]]] = defaultdict(list); by_vid: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row); by_vid[row["video"]].append(row)
    fold_unsafe = float(np.mean([int(r["unsafe_flip"]) for r in rows])) if rows else 0.0
    return {
        "queries": len(rows), "r1": _mean(rows, "r1"), "r5": _mean(rows, "r5"), "map": _mean(rows, "map"),
        "raw_r1": _mean(rows, "raw_r1"), "raw_r5": _mean(rows, "raw_r5"), "raw_map": _mean(rows, "raw_map"),
        "hard_negative_gap": _mean(rows, "hard_negative_gap"), "raw_hard_negative_gap": _mean(rows, "raw_hard_negative_gap"),
        "category_macro_r1": float(np.mean([_mean(x, "r1") for x in by_cat.values()])) if by_cat else 0.0,
        "video_macro_r1": float(np.mean([_mean(x, "r1") for x in by_vid.values()])) if by_vid else 0.0,
        "category_count": len(by_cat), "video_count": len(by_vid),
        "unsafe_flip_count": int(sum(int(r["unsafe_flip"]) for r in rows)),
        "unsafe_flip_micro_rate": fold_unsafe,
        "top1_change_count": int(sum(int(r["top1_changed"]) for r in rows)),
        "top1_change_rate": float(sum(int(r["top1_changed"]) for r in rows) / max(len(rows), 1)),
        "per_query": rows,
    }


def aggregate_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("r1", "r5", "map", "raw_r1", "raw_r5", "raw_map", "hard_negative_gap", "raw_hard_negative_gap", "category_macro_r1", "video_macro_r1")
    out = {k: float(np.mean([m[k] for m in fold_metrics])) if fold_metrics else 0.0 for k in keys}
    total = sum(int(m["queries"]) for m in fold_metrics)
    out.update({
        "folds": len(fold_metrics), "queries": total,
        "unsafe_flip_count": int(sum(m["unsafe_flip_count"] for m in fold_metrics)),
        "unsafe_flip_micro_rate": float(sum(m["unsafe_flip_count"] for m in fold_metrics) / max(total, 1)),
        "unsafe_flip_fold_macro_rate": float(np.mean([m["unsafe_flip_micro_rate"] for m in fold_metrics])) if fold_metrics else 0.0,
        "top1_change_count": int(sum(m["top1_change_count"] for m in fold_metrics)),
        "top1_change_micro_rate": float(sum(m["top1_change_count"] for m in fold_metrics) / max(total, 1)),
        "top1_change_fold_macro_rate": float(np.mean([m["top1_change_rate"] for m in fold_metrics])) if fold_metrics else 0.0,
    })
    return out
