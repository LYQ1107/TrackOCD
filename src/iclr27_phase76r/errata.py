"""Pure helpers for the Phase75 contract errata.

The helpers deliberately accept already materialised metric rows.  Labels,
identities, and evaluator outcomes are never part of a model tensor here.
"""
from __future__ import annotations

from typing import Any, Iterable


def _delta(row: dict[str, Any], metric: str) -> float:
    if metric == "r1":
        return float(row["pairwise_r1"] - row["raw_r1"])
    if metric == "map":
        return float(row["pairwise_map"] - row["raw_map"])
    if metric == "hard_gap":
        return float(row["pairwise_hard_gap"] - row["raw_hard_gap"])
    raise KeyError(metric)


def correct_teacher_authorization(
    global_p16_folds: Iterable[dict[str, Any]],
    legal_p16_folds: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Re-evaluate the historical teacher rule with its intended global guard.

    The old helper accidentally applied ``no_large_r1_drop`` to legal rows.
    The corrected contract checks all global folds, while retaining the other
    registered teacher conditions for transparency.
    """
    global_rows = list(global_p16_folds)
    legal_rows = list(legal_p16_folds)
    old = all(float(r["delta_r1"]) >= -0.02 for r in legal_rows)
    global_bad = [
        {"fold": int(r["fold"]), "delta_r1": float(r["delta_r1"])}
        for r in global_rows
        if float(r["delta_r1"]) < -0.02
    ]
    corrected = len(global_bad) == 0
    return {
        "historical_phase75d_teacher_authorization": "ERRATUM",
        "old_result": bool(old),
        "corrected_result": bool(corrected),
        "reason": "global folds fall below -0.02" if global_bad else "no global fold falls below -0.02",
        "global_bad_folds": global_bad,
        "legal_rows_checked": len(legal_rows),
        "global_rows_checked": len(global_rows),
    }


def checkpoint_in_safe_window(row: dict[str, Any]) -> bool:
    """Diagnostic Phase76R window; this never authorizes a model gate."""
    return bool(
        int(row.get("global_unsafe", 0)) == 0
        and int(row.get("legal_unsafe", 0)) == 0
        and float(row.get("global_delta_r1", 0.0)) >= -0.005
        and float(row.get("global_delta_map", 0.0)) >= -0.002
        and float(row.get("legal_delta_r1", 0.0)) > 0.0
        and float(row.get("legal_delta_map", 0.0)) > 0.0
        and float(row.get("mean_raw_adapt_cosine", 0.0)) >= 0.98
    )

