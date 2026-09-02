"""Track-level (not per-row mixed) evaluator-only alignment."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .failure_taxonomy import failure_record
from .prefix_contract import PREFIXES, get_visible_source_rows, get_visible_target_rows


def _xyxy(row: Mapping[str, Any]) -> list[float] | None:
    try:
        value = row.get("bbox_xyxy", "")
        if isinstance(value, str): value = value.strip("[]").split(",")
        vals = [float(x) for x in value]
        return vals if len(vals) == 4 else None
    except (TypeError, ValueError): return None


def _qbox(q: Mapping[str, Any]) -> list[float] | None:
    try:
        b = [float(x) for x in q.get("bbox", [])]
        return [b[0], b[1], b[0] + max(0., b[2]), b[1] + max(0., b[3])] if len(b) == 4 else None
    except (TypeError, ValueError): return None


def iou(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if a is None or b is None: return 0.0
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); inter = max(0., ix2-ix1)*max(0., iy2-iy1)
    aa = max(0., a[2]-a[0])*max(0., a[3]-a[1]); bb = max(0., b[2]-b[0])*max(0., b[3]-b[1]); den = aa+bb-inter
    return inter/den if den > 0 else 0.0


def align_tracklet(event: Mapping[str, Any], role: str, tracklet_key: str, prefix: int, event_rows: Sequence[Mapping[str, Any]], q0_by_image: Mapping[str, Sequence[Mapping[str, Any]]], *, source_file: str) -> dict[str, Any]:
    # Source visibility is the complete per-tracklet registration sequence;
    # only target rows are truncated by the evaluator prefix.
    selected = list(event_rows) if role == "source" else list(event_rows[: min(int(prefix), len(event_rows))]); per_track: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": [], "ious": [], "frames": []}); failures = []
    for row in selected:
        ik = row.get("canonical_image_key") or str(row.get("image_id", "")); candidates = list(q0_by_image.get(str(ik), ())); eb = _xyxy(row)
        if not candidates:
            failures.append(failure_record(str(event.get("event_key")), role, tracklet_key, prefix, video_key=row.get("canonical_video_key"), image_key=str(ik), source_file=source_file, source_ref=str(row.get("row_key", "")), code="ASSET_NOT_PRESENT_IN_EXISTING_Q0", reason="no Q0 candidate for canonical image", available={"candidate_count": 0}, missing=["q0_image_mapping"], recoverable=True)); continue
        for q in candidates:
            qk = f"v{q.get('video_id')}:p{q.get('track_id')}"; qi = iou(eb, _qbox(q)); per_track[qk]["rows"].append({"event_row_key": row.get("row_key"), "q0_iou": qi, "q0_score": q.get("score"), "q0_bbox_xyxy": _qbox(q), "image_key": ik}); per_track[qk]["ious"].append(qi); per_track[qk]["frames"].append(row.get("frame_id"))
    eligible = [k for k, v in per_track.items() if any(float(x) >= .5 for x in v["ious"])]
    classification = "UNMATCHED" if len(eligible) == 0 else ("UNIQUE_MAPPING" if len(eligible) == 1 else "AMBIGUOUS")
    if len(eligible) > 1: failures.append(failure_record(str(event.get("event_key")), role, tracklet_key, prefix, video_key=None, image_key=None, source_file=source_file, source_ref=None, code="MULTIPLE_ELIGIBLE_PHYSICAL_TRACKS", reason="more than one Q0 physical track has IoU>=0.5; no score/ID tie-break", available={"eligible_tracks": eligible}, missing=[], recoverable=False))
    if not selected and not failures: failures.append(failure_record(str(event.get("event_key")), role, tracklet_key, prefix, video_key=None, image_key=None, source_file=source_file, source_ref=None, code="MISSING_EVENT_TRACKLET_ROWS", reason="tracklet has no visible rows", available={}, missing=["event_rows"], recoverable=True))
    return {"event_key": str(event.get("event_key")), "fold": int(event.get("fold", -1)), "kind": event.get("kind"), "role": role, "event_tracklet_key": tracklet_key, "prefix": int(prefix), "selected_event_rows": len(selected), "candidate_physical_tracks": {k: {"row_count": len(v["rows"]), "max_iou": max(v["ious"]) if v["ious"] else None, "mean_iou": sum(v["ious"])/len(v["ious"]) if v["ious"] else None, "frames": v["frames"], "rows": v["rows"]} for k, v in per_track.items()}, "eligible_physical_tracks": eligible, "mapping_classification": classification, "failure_records": failures}
