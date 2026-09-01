from __future__ import annotations


def emit(sample_id, stream_order, decision, known_id=None, virtual_id=None):
    """Canonical TrackOCD-v1.0 label-space output."""
    if decision == "known" and known_id is not None:
        return {
            "sample_id": sample_id,
            "stream_order": stream_order,
            "prediction_type": "known",
            "semantic_category_id": int(known_id),
            "virtual_category_id": None,
        }
    if decision == "novel" and virtual_id is not None:
        return {
            "sample_id": sample_id,
            "stream_order": stream_order,
            "prediction_type": "novel",
            "semantic_category_id": None,
            "virtual_category_id": int(virtual_id),
        }
    return {
        "sample_id": sample_id,
        "stream_order": stream_order,
        "prediction_type": "unresolved",
        "semantic_category_id": None,
        "virtual_category_id": None,
    }
