"""Isolated model-facing null exporter and evaluator adapter."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .contracts import null_prediction, track_key


def null_policy_records(q0_rows: Iterable[Mapping[str, Any]], limit: int = 1000) -> list[dict[str, Any]]:
    """Produce a bounded plumbing sample from Q0 only.

    It intentionally does not consume event manifests, labels, categories or
    alignment results.  The bounded sample is not an OCD performance result;
    it demonstrates the legal no-support fallback until a future semantic
    model is explicitly authorized.
    """
    positions: dict[str, int] = defaultdict(int)
    out: list[dict[str, Any]] = []
    for row in q0_rows:
        if len(out) >= limit:
            break
        try:
            key = track_key(row["video_id"], row["track_id"])
            image_id = int(row["image_id"])
        except (KeyError, TypeError, ValueError):
            continue
        pos = positions[key]
        positions[key] += 1
        out.append(null_prediction(key, image_id, pos))
    return out


def evaluator_null_row(alignment: Mapping[str, Any]) -> dict[str, Any]:
    """Post-hoc adapter row; event labels remain evaluator metadata only."""
    return {
        "event_key": alignment.get("event_key"),
        "fold": alignment.get("fold"),
        "kind": alignment.get("kind"),
        "role": alignment.get("role"),
        "prefix": alignment.get("prefix"),
        "adapter_status": "CONTRACT_NULL_POLICY",
        "prediction_type": "unresolved",
        "semantic_category_id": None,
        "virtual_category_id": None,
        "action": "DEFER",
        "uncertainty": 1.0,
        "metric_status": "NOT_RUN_NO_SEMANTIC_MODEL",
        "alignment_ref": {"mapping_layer": alignment.get("mapping_layer"), "mapping_method": alignment.get("mapping_method")},
    }
