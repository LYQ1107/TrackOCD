"""Small, auditable Phase73 contract primitives.

These functions are deliberately free of category/ID semantics.  Track and
video identifiers are accepted only as bookkeeping keys and never returned as
feature vectors.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

PREFIXES = (1, 2, 4, 8, 16)
RELIABLE_RULE = "assigned == 1 and transformed_iou >= 0.5"
MODEL_NULL_POLICY = {
    "prediction_type": "unresolved",
    "semantic_category_id": None,
    "virtual_category_id": None,
    "action": "DEFER",
    "support": None,
    "uncertainty": 1.0,
}


def track_key(video_id: Any, track_id: Any) -> str:
    """Canonical bookkeeping key used by the legacy event/CSV stream."""
    return f"v{int(video_id)}:p{int(track_id)}"


def row_key(row: Mapping[str, Any]) -> str:
    """Five-field corrected-CSV row key (the historical order is explicit)."""
    if row.get("row_key"):
        return str(row["row_key"])
    fields = ("video_id", "frame_id", "proposal_local_id", "track_id", "image_id")
    return ":".join(str(row[k]) for k in fields)


def finite_box(box: Sequence[Any]) -> bool:
    try:
        vals = [float(x) for x in box]
    except (TypeError, ValueError):
        return False
    return len(vals) == 4 and all(math.isfinite(x) for x in vals)


def xywh_to_xyxy(box: Sequence[Any]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in box[:4]]
    return (x, y, x + max(w, 0.0), y + max(h, 0.0))


def xyxy_iou(a: Sequence[Any], b: Sequence[Any]) -> float:
    if not finite_box(a) or not finite_box(b):
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + ab - inter
    return inter / den if den > 0.0 else 0.0


def assigned(row: Mapping[str, Any]) -> bool:
    return str(row.get("assigned", "0")).lower() in {"1", "true", "yes"}


def event_row_reliable(row: Mapping[str, Any], threshold: float = 0.5) -> bool:
    try:
        value = float(row.get("row_iou", 0.0))
    except (TypeError, ValueError):
        value = 0.0
    return assigned(row) and value >= threshold


def no_forbidden_model_fields(record: Mapping[str, Any]) -> list[str]:
    forbidden = {
        "category", "category_id", "gt_category_id", "gt_category_id_common",
        "semantic_category_id", "virtual_category_id", "event_key", "event_kind",
        "expected_first_commit", "positive_label", "negative_label", "label",
        "source_true_pair", "target_true_pair", "row_iou", "event_iou",
        "physical_id", "semantic_id", "text", "future_frame", "future_track",
    }
    return sorted(k for k in record if k in forbidden and record[k] not in (None, "", False))


def null_prediction(bookkeeping_track: str, image_id: Any, causal_position: int) -> dict[str, Any]:
    """CONTRACT_NULL_POLICY record; no semantic inference is claimed."""
    out: dict[str, Any] = {
        "physical_track_key": str(bookkeeping_track),
        "image_id": int(image_id) if image_id is not None else None,
        "causal_position": int(causal_position),
        "representation": None,
        "representation_dim": 768,
        "representation_status": "UNAVAILABLE_PHASE73_NO_SEMANTIC_MODEL",
        "contract_policy": "CONTRACT_NULL_POLICY",
    }
    out.update(MODEL_NULL_POLICY)
    return out
