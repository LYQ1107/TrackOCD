"""Track-level Q0 alignment with explicit joint reliability and fragmentation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .prefix_contract import PREFIXES, sorted_track_rows


def xyxy(value: Any) -> list[float] | None:
    try:
        if isinstance(value, str):
            value = value.strip("[]").split(",")
        result = [float(x) for x in value]
        return result if len(result) == 4 else None
    except (TypeError, ValueError):
        return None


def q0_xywh(value: Any) -> list[float] | None:
    try:
        result = [float(x) for x in value]
        return [result[0], result[1], result[0] + max(0.0, result[2]), result[1] + max(0.0, result[3])] if len(result) == 4 else None
    except (TypeError, ValueError):
        return None


def iou(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def _event_reliable(row: Mapping[str, Any]) -> bool:
    try:
        return int(row.get("assigned", 0)) == 1 and float(row.get("row_iou", 0.0) or 0.0) >= 0.5
    except (TypeError, ValueError):
        return False


class TrackletAligner:
    def __init__(self, event_asset_by_image: Mapping[int, Mapping[str, Any]], physical_index: Any) -> None:
        self.event_asset_by_image = event_asset_by_image
        self.physical_index = physical_index

    def align(self, event: Mapping[str, Any], role: str, tracklet_key: str, prefix: int, event_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        ordered = sorted_track_rows(event_rows)
        selected = ordered if role == "source" else ordered[: min(int(prefix), len(ordered))]
        candidates: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": [], "frames": [], "q0_ious": [], "joint_frames": []})
        available_rows = 0
        for event_row in selected:
            try:
                image_id = int(event_row.get("image_id"))
            except (TypeError, ValueError):
                image_id = -1
            asset = self.event_asset_by_image.get(image_id, {})
            content_key = str(asset.get("content_asset_key", ""))
            q0_rows = self.physical_index.lookup_image(content_key)
            available_rows += len(q0_rows)
            event_box = xyxy(event_row.get("bbox_xyxy"))
            frame = event_row.get("frame_id")
            for candidate in q0_rows:
                physical_key = f"v{candidate.get('video_id')}:p{candidate.get('track_id')}"
                q0_box = q0_xywh(candidate.get("bbox", []))
                q0_iou = iou(event_box, q0_box)
                joint = _event_reliable(event_row) and q0_iou >= 0.5
                item = {"event_row_key": event_row.get("row_key"), "frame_id": frame, "image_id": image_id, "q0_image_id": candidate.get("image_id"), "q0_iou": q0_iou, "event_reliable": _event_reliable(event_row), "joint_reliable": joint, "candidate_order": candidate.get("candidate_order")}
                candidates[physical_key]["rows"].append(item)
                candidates[physical_key]["frames"].append(frame)
                candidates[physical_key]["q0_ious"].append(q0_iou)
                if joint:
                    candidates[physical_key]["joint_frames"].append(frame)
        eligible = [key for key, value in candidates.items() if value["joint_frames"]]
        segments = []
        for key in eligible:
            frames = [int(x) for x in candidates[key]["joint_frames"] if str(x).lstrip("-").isdigit()]
            if frames:
                segments.append({"track": key, "first_frame": min(frames), "last_frame": max(frames)})
        overlap = False
        if len(eligible) > 1:
            frame_sets = [set(candidates[key]["joint_frames"]) for key in eligible]
            overlap = any(frame_sets[i] & frame_sets[j] for i in range(len(frame_sets)) for j in range(i + 1, len(frame_sets)))
        if not eligible:
            classification = "UNMATCHED"
        elif len(eligible) == 1:
            classification = "UNIQUE_MAPPING"
        elif overlap:
            classification = "AMBIGUOUS_OVERLAP"
        else:
            classification = "PHYSICAL_FRAGMENTATION"
        max_iou = max((max(value["q0_ious"]) for value in candidates.values() if value["q0_ious"]), default=None)
        return {
            "event_key": str(event.get("event_key", "")),
            "fold": int(event.get("fold", -1)),
            "kind": str(event.get("kind", "")),
            "role": role,
            "event_tracklet_key": str(tracklet_key),
            "prefix": int(prefix),
            "selected_event_rows": len(selected),
            "q0_candidate_count": available_rows,
            "q0_max_iou": max_iou,
            "eligible_physical_tracks": eligible,
            "mapping_classification": classification,
            "segments": segments,
            "candidate_physical_tracks": {
                key: {
                    "row_count": len(value["rows"]),
                    "max_iou": max(value["q0_ious"]) if value["q0_ious"] else None,
                    "joint_reliable_frames": value["joint_frames"],
                    "rows": value["rows"],
                }
                for key, value in candidates.items()
            },
            "reliability_rule": "event assigned == 1 AND event row_iou >= 0.5 AND q0 IoU >= 0.5",
        }
