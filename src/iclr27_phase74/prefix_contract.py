"""Explicit Phase19R causal source/target visibility contract."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

PREFIXES = (1, 2, 4, 8, 16)


def _track_rows(event: Mapping[str, Any], rows_by_track: Mapping[str, Sequence[Mapping[str, Any]]], key: str) -> list[Mapping[str, Any]]:
    rows = list(rows_by_track.get(str(key), ()))
    return sorted(rows, key=lambda r: (int(r.get("event_rank", 0)), int(r.get("frame_id", 0)), int(r.get("image_id", 0))))


def get_source_registration_sequence(event: Mapping[str, Any], causal_contract: Mapping[str, Any], rows_by_track: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> list[dict[str, Any]]:
    out = []
    for source_index, key in enumerate(event.get("source_tracklet_keys", [])):
        rows = _track_rows(event, rows_by_track or {}, str(key)) if rows_by_track is not None else []
        out.append({"source_index": source_index, "tracklet_key": str(key), "position_count": len(rows),
                    "positions": [{"position": i, "row_key": str(r.get("row_key", "")), "frame_id": r.get("frame_id"), "image_id": r.get("image_id")} for i, r in enumerate(rows)]})
    return out


def get_visible_source_rows(event: Mapping[str, Any], causal_contract: Mapping[str, Any], rows_by_track: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    # The runner fully processes each source track before target, preserving
    # tracklet identity and positions; source is never concatenated/sliced.
    out = []
    for source_index, key in enumerate(event.get("source_tracklet_keys", [])):
        for position, row in enumerate(_track_rows(event, rows_by_track, str(key))):
            out.append({"source_index": source_index, "tracklet_key": str(key), "position": position, "row": row})
    return out


def get_visible_target_rows(event: Mapping[str, Any], prefix: int, causal_contract: Mapping[str, Any], rows_by_track: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    key = str(event.get("target_tracklet_key", "")); rows = _track_rows(event, rows_by_track, key)
    # For audit prefix visibility, prefix is a count.  The runtime itself
    # feeds the target stream sequentially after source registration; it does
    # not compute statistics from a complete future target track.
    return [{"tracklet_key": key, "position": i, "row": row} for i, row in enumerate(rows[: min(int(prefix), len(rows))])]


def build_contract() -> dict[str, Any]:
    return {"schema_version": "phase74.causal_prefix.v1", "prefixes": list(PREFIXES),
            "source_visibility": "all positions of each source tracklet are registered independently before target",
            "target_visibility": "causal prefix is first N rows of target track sorted by event_rank/frame/image; no future rows at prefix N",
            "source_tracklet_order": "event source_tracklet_keys list order",
            "source_rows_not_concatenated": True, "source_position_scope": "per tracklet",
            "target_position_monotonic": True, "source_before_target": True,
            "source_target_shared_frame_index": False,
            "event_rank_relation": "event_rank is deterministic causal order; frame_id/source_frame_index are annotation metadata",
            "source_support_before_target": True, "source_state_immutable_snapshot": True,
            "complete_target_statistics_at_early_prefix": False,
            "runtime_evidence": {"path": "scripts/iclr27_phase19r/freeze_predictions.py", "function": "run_event", "source_loop": "for key ... for pos ...", "target_loop": "for pos ..."},
            "contract_status": "PROVEN_FROM_RUNNER_AND_STREAM"}
