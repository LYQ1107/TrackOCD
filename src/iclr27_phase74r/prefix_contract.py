"""Causal source registration and target-prefix semantics."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

PREFIXES = (1, 2, 4, 8, 16)


def sorted_track_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(rows, key=lambda row: (int(row.get("event_rank", 0)), int(row.get("frame_id", 0)), int(row.get("image_id", 0))))


def source_rows(event: Mapping[str, Any], rows_by_track: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_index, key in enumerate(event.get("source_tracklet_keys", [])):
        for position, row in enumerate(sorted_track_rows(rows_by_track.get(str(key), []))):
            output.append({"source_index": source_index, "tracklet_key": str(key), "position": position, "row": dict(row)})
    return output


def target_rows(event: Mapping[str, Any], prefix: int, rows_by_track: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    key = str(event.get("target_tracklet_key", ""))
    rows = sorted_track_rows(rows_by_track.get(key, []))[: int(prefix)]
    return [{"tracklet_key": key, "position": i, "row": dict(row)} for i, row in enumerate(rows)]


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "phase74r.causal_prefix.v1",
        "prefixes": list(PREFIXES),
        "source_visibility": "all positions of each source tracklet are registered independently before target",
        "source_rows_not_concatenated": True,
        "source_position_scope": "per tracklet",
        "target_visibility": "first N target rows sorted by event_rank/frame/image; no future rows",
        "target_position_monotonic": True,
        "source_before_target": True,
        "source_state_immutable_snapshot": True,
        "complete_target_statistics_at_early_prefix": False,
        "physical_id_is_bookkeeping_only": True,
        "runtime_evidence": {"path": "scripts/iclr27_phase19r/freeze_predictions.py", "function": "run_event"},
    }
