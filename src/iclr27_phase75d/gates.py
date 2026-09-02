"""Pre-registered Phase75D strict gates and pairwise teacher signal."""
from __future__ import annotations

from typing import Any


def gate_rows(folds: list[dict[str, Any]], section: str) -> list[dict[str, Any]]:
    rows = []
    for f in folds:
        raw = f["metrics"]["raw"]; pair = f["metrics"]["pairwise"]
        rows.append({
            "fold": f["fold"], "raw_r1": raw["r1"], "pairwise_r1": pair["r1"], "delta_r1": pair["r1"] - raw["r1"],
            "raw_map": raw["map"], "pairwise_map": pair["map"], "delta_map": pair["map"] - raw["map"],
            "raw_hard_gap": raw["hard_negative_gap"], "pairwise_hard_gap": pair["hard_negative_gap"],
            "delta_hard_gap": pair["hard_negative_gap"] - raw["hard_negative_gap"],
            "unsafe_flip": pair["unsafe_flip_count"],
            "substantial": bool(pair["r1"] - raw["r1"] >= 0.02 and pair["map"] - raw["map"] >= 0.01),
            "directional": bool(pair["r1"] > raw["r1"] and pair["map"] > raw["map"]),
            "section": section,
        })
    return rows


def strict_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    substantial = sum(int(r["substantial"]) for r in rows)
    return {
        "thresholds": {"delta_r1": 0.02, "delta_map": 0.01, "minimum_folds": 3, "unsafe_flip": 0},
        "folds_substantial": substantial,
        "folds_directional": sum(int(r["directional"]) for r in rows),
        "unsafe_flip_count": sum(int(r["unsafe_flip"]) for r in rows),
        "hard_gap_non_worse": all(r["delta_hard_gap"] >= -1e-7 for r in rows),
        "pass": substantial >= 3 and sum(int(r["unsafe_flip"]) for r in rows) == 0 and all(r["delta_hard_gap"] >= -1e-7 for r in rows),
        "rows": rows,
    }


def pairwise_teacher_signal(global_aggregate: dict[str, Any], legal_aggregate: dict[str, Any], legal_folds: list[dict[str, Any]]) -> dict[str, Any]:
    legal_gap_positive_folds = sum(int(f["metrics"]["pairwise"]["hard_negative_gap"] > f["metrics"]["raw"]["hard_negative_gap"]) for f in legal_folds)
    no_large_r1_drop = all(float(f["metrics"]["pairwise"]["r1"] - f["metrics"]["raw"]["r1"]) >= -0.02 for f in legal_folds)
    value = bool(
        legal_aggregate["map"] - legal_aggregate["raw_map"] > 0
        and legal_gap_positive_folds >= 3
        and global_aggregate["map"] - global_aggregate["raw_map"] >= 0
        and no_large_r1_drop
    )
    return {
        "legal_delta_map": legal_aggregate["map"] - legal_aggregate["raw_map"],
        "legal_gap_positive_folds": legal_gap_positive_folds,
        "global_delta_map": global_aggregate["map"] - global_aggregate["raw_map"],
        "no_fold_delta_r1_below_minus_0.02": no_large_r1_drop,
        "signal": value,
        "decision": "P75D_PAIRWISE_SIGNAL_AUTHORIZE_P75E" if value else "P75D_NO_PAIRWISE_SIGNAL",
    }
