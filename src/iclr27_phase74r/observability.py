"""Null-safe pre-replay observability tables with exact denominators."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .prefix_contract import PREFIXES


def _metric(numerator: int | None, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator, "value": None if numerator is None else numerator / max(1, denominator)}


def null_records(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        for role in ("source", "target"):
            for prefix in PREFIXES:
                rows.append({
                    "event_key": str(event.get("event_key", "")),
                    "fold": int(event.get("fold", -1)),
                    "kind": str(event.get("kind", "")),
                    "role": role,
                    "prefix": prefix,
                    "status": "NOT_AVAILABLE_Q0_NOT_REPLAYED",
                    "asset_available": None,
                    "q0_processed": None,
                    "candidate_observed": None,
                    "joint_reliable": None,
                    "unique_mapping": None,
                    "ambiguous_overlap": None,
                    "fragmentation": None,
                    "no_detection": None,
                    "low_iou": None,
                    "event_row_unreliable": None,
                    "q0_candidate_count": None,
                    "q0_max_iou": None,
                    "metric_status": "NOT_AVAILABLE_Q0_NOT_REPLAYED",
                })
    return rows


def _aggregate(rows: list[dict[str, Any]], unique_keys: bool = True) -> dict[str, Any]:
    keys = {str(row["event_key"]) for row in rows} if unique_keys else set()
    den = len(keys) if unique_keys else len(rows)
    # Before replay every binary observation is genuinely unknown, never zero.
    return {key: _metric(None, den) for key in (
        "asset_available", "q0_processed", "candidate_observed", "joint_reliable",
        "unique_mapping", "ambiguous_overlap", "fragmentation", "no_detection",
        "low_iou", "event_row_unreliable")}


def build_tables(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = null_records(events)
    def subset(**filters: Any) -> list[dict[str, Any]]:
        return [row for row in rows if all(row.get(k) == v for k, v in filters.items())]
    by_prefix = {str(prefix): {role: _aggregate(subset(prefix=prefix, role=role)) for role in ("source", "target")} for prefix in PREFIXES}
    by_role = {role: _aggregate(subset(role=role)) for role in ("source", "target")}
    by_polarity = {polarity: {role: _aggregate(subset(role=role, kind=polarity)) for role in ("source", "target")} for polarity in ("positive", "negative")}
    by_fold: dict[str, Any] = {}
    for fold in range(4):
        by_fold[str(fold)] = {
            "total_events": len({row["event_key"] for row in events if int(row.get("fold", -1)) == fold}),
            "positive_events": len({row["event_key"] for row in events if int(row.get("fold", -1)) == fold and str(row.get("kind")) == "positive"}),
            "negative_events": len({row["event_key"] for row in events if int(row.get("fold", -1)) == fold and str(row.get("kind")) == "negative"}),
            "source": _aggregate(subset(fold=fold, role="source")),
            "target": _aggregate(subset(fold=fold, role="target")),
        }
    # Pairing is defined on (event_key, prefix), but remains null before replay.
    pair_keys = {(row["event_key"], row["prefix"]) for row in rows}
    summary = {
        "schema_version": "phase74r.observability.v1",
        "status": "NOT_AVAILABLE_Q0_NOT_REPLAYED",
        "positive_events": len({row["event_key"] for row in events if str(row.get("kind")) == "positive"}),
        "negative_events": len({row["event_key"] for row in events if str(row.get("kind")) == "negative"}),
        "total_events": len({row["event_key"] for row in events}),
        "records": len(rows),
        "prefixes": list(PREFIXES),
        "reliable_rule": "event assigned == 1 AND transformed/event row IoU >= 0.5 AND Q0 candidate IoU >= 0.5",
        "both_reliable_pairing_key": "(event_key, prefix)",
        "both_reliable_denominator": len(pair_keys),
        "by_prefix": by_prefix,
        "by_role": by_role,
        "by_polarity": by_polarity,
        "by_fold": by_fold,
        "failure_reasons": Counter(row["status"] for row in rows),
        "not_detector_failure": True,
    }
    return rows, summary
