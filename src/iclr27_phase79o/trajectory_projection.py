from __future__ import annotations

from collections import defaultdict
from typing import Any


def _valid_box(box: list[float] | None) -> list[float] | None:
    if box is None or len(box) != 4: return None
    x1, y1, x2, y2 = [float(v) for v in box]
    if x2 <= x1 or y2 <= y1: return None
    return [x1, y1, x2, y2]


def _project(previous: dict[str, Any], older: dict[str, Any] | None, frame_id: int, *, max_gap: int) -> list[float] | None:
    box = _valid_box(previous.get("bbox_xyxy"));
    if box is None: return None
    gap = int(frame_id) - int(previous.get("frame_id", frame_id))
    if gap <= 0 or gap > int(max_gap): return None
    if older is None or _valid_box(older.get("bbox_xyxy")) is None:
        return box
    old = _valid_box(older["bbox_xyxy"]); dt = max(int(previous.get("frame_id", 0)) - int(older.get("frame_id", 0)), 1)
    velocity = [(box[i] - old[i]) / dt for i in range(4)]
    projected = [box[i] + velocity[i] * gap for i in range(4)]
    x1, y1, x2, y2 = projected
    if x2 <= x1 or y2 <= y1: return box
    return projected


def build_causal_projection_lookup(native_rows: list[dict[str, Any]], event_keys: set[tuple[int, int]], *, max_gap: int = 2) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], dict[str, Any]]:
    """Augment event-frame candidates using only strictly earlier track rows.

    Existing candidates are never removed.  A synthetic candidate is a
    constant-velocity projection from the latest one or two observations of a
    physical track.  The function is intended to be called on a time-sorted
    stream; it never reads a row at or after the requested frame when making a
    projection.
    """
    by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    histories: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in native_rows:
        if row.get("bbox_xyxy") is None: continue
        key = (int(row["video_id"]), int(row["image_id"])); by_key[key].append(dict(row))
        track = (int(row["video_id"]), int(row["physical_track_id"]))
        histories[track].append(row); by_video[int(row["video_id"])].append(row)
    for rows in histories.values(): rows.sort(key=lambda x: (int(x.get("frame_id", 0)), int(x.get("image_id", 0))))
    for rows in by_video.values(): rows.sort(key=lambda x: (int(x.get("frame_id", 0)), int(x.get("image_id", 0))))
    added = 0; keys_with_projection = 0
    for video, image_id in sorted(event_keys):
        current_rows = by_key.get((video, image_id), [])
        current_frame_values = [int(row.get("frame_id", 0)) for row in current_rows]
        if not current_frame_values: continue
        frame_id = min(current_frame_values)
        augmented = list(current_rows); existing_tracks = {int(row["physical_track_id"]) for row in current_rows}
        for (track_video, track_id), hist in histories.items():
            if track_video != video: continue
            prior = [row for row in hist if int(row.get("frame_id", 0)) < frame_id]
            if not prior: continue
            previous = prior[-1]; older = prior[-2] if len(prior) >= 2 else None
            box = _project(previous, older, frame_id, max_gap=max_gap)
            if box is None: continue
            # Do not create a duplicate track candidate when a current row
            # already exists; the raw candidate remains untouched.
            if track_id in existing_tracks: continue
            score = float(previous.get("base_score") or 0.0) * (0.8 ** max(frame_id - int(previous.get("frame_id", frame_id)), 1))
            augmented.append({
                "video_id": video, "image_id": image_id, "frame_id": frame_id,
                "physical_track_id": track_id, "parent_physical_track_id": track_id,
                "proposal_local_id": -1, "candidate_rank": 100000 + added,
                "bbox_xyxy": box, "base_score": score,
                "lifecycle": "causal_projection", "hit_count": int(previous.get("hit_count") or 0),
                "disappear_time": int(previous.get("disappear_time") or 0),
                "score_mode": "causal_projection", "source": "strictly_prior_velocity",
            }); added += 1
        if len(augmented) > len(current_rows): keys_with_projection += 1
        by_key[(video, image_id)] = augmented
    return dict(by_key), {"native_rows": len(native_rows), "event_image_keys": len(event_keys), "synthetic_candidates": added, "event_keys_with_projection": keys_with_projection, "max_gap": int(max_gap), "causal_rule": "strictly prior frame_id; latest two boxes only; existing candidates retained"}
