"""Evaluator-only temporal alignment between event CSV rows and Q0 TAO rows."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .contracts import PREFIXES, event_row_reliable, track_key, xywh_to_xyxy, xyxy_iou


def parse_event_rows(events: Iterable[Mapping[str, Any]], csv_rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], set[int], set[int]]:
    """Index only rows belonging to event source/target tracks (audit layer)."""
    wanted_tracks: set[str] = set()
    wanted_videos: set[int] = set()
    wanted_rows: set[str] = set()
    for event in events:
        wanted_videos.update((int(event["source_video"]), int(event["target_video"])))
        wanted_tracks.update(str(x) for x in event.get("source_tracklet_keys", []))
        wanted_tracks.add(str(event.get("target_tracklet_key", "")))
        wanted_rows.update(str(x) for x in event.get("target_row_keys", []))
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row0 in csv_rows:
        row = dict(row0)
        key = track_key(row["video_id"], row["track_id"])
        if key not in wanted_tracks:
            continue
        # Keep evaluator fields in this index, never pass this object to a model.
        by_track[key].append(row)
    for rows in by_track.values():
        rows.sort(key=lambda r: (int(r.get("event_rank", 0)), int(r.get("frame_id", 0)), int(r.get("image_id", 0))))
    return by_track, wanted_videos, wanted_rows


def q0_index_for_images(q0_rows: Iterable[Mapping[str, Any]], image_ids: set[int]) -> tuple[dict[int, list[dict[str, Any]]], Counter, set[str]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    count = Counter()
    tracks: set[str] = set()
    for row0 in q0_rows:
        try:
            image_id = int(row0["image_id"])
        except (KeyError, TypeError, ValueError):
            continue
        count["records_total"] += 1
        if image_id not in image_ids:
            continue
        row = dict(row0)
        by_image[image_id].append(row)
        count["records_needed_images"] += 1
        try:
            tracks.add(track_key(row["video_id"], row["track_id"]))
        except (KeyError, TypeError, ValueError):
            pass
    return by_image, count, tracks


def _best_for_row(row: Mapping[str, Any], q0_candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        event_box = [float(x) for x in str(row.get("bbox_xyxy", "[]")).strip("[]").split(",")]
    except (TypeError, ValueError):
        event_box = []
    if len(event_box) != 4:
        event_box = []
    best = None
    for q in q0_candidates:
        try:
            qbox = xywh_to_xyxy(q["bbox"])
            score = float(q.get("score", 0.0))
            qiou = xyxy_iou(event_box, qbox)
        except (KeyError, TypeError, ValueError):
            continue
        cand = {"q0_track_key": track_key(q.get("video_id"), q.get("track_id")), "q0_iou": qiou, "q0_score": score, "q0_bbox_xyxy": qbox}
        if best is None or (qiou, score) > (best["q0_iou"], best["q0_score"]):
            best = cand
    if best is None:
        return {"q0_candidate_count": 0, "q0_best_iou": None, "q0_best_score": None, "q0_track_key": None}
    best["q0_candidate_count"] = len(q0_candidates)
    return best


def align_role(rows: Sequence[Mapping[str, Any]], q0_by_image: Mapping[int, Sequence[Mapping[str, Any]]], prefix: int) -> dict[str, Any]:
    selected = list(rows[: min(prefix, len(rows))])
    matches: list[dict[str, Any]] = []
    missing = 0
    reliable_event_rows = 0
    for row in selected:
        try:
            image_id = int(row["image_id"])
        except (KeyError, TypeError, ValueError):
            image_id = None
        candidates = q0_by_image.get(image_id, ()) if image_id is not None else ()
        best = _best_for_row(row, candidates)
        if not candidates:
            missing += 1
        if event_row_reliable(row):
            reliable_event_rows += 1
        matches.append({
            "image_id": image_id,
            "frame_id": int(row["frame_id"]) if str(row.get("frame_id", "")).isdigit() else None,
            "event_row_key": str(row.get("row_key", "")),
            "event_assigned": str(row.get("assigned", "0")),
            "event_row_iou": float(row.get("row_iou", 0.0) or 0.0),
            "event_row_reliable": event_row_reliable(row),
            "q0": best,
        })
    q0_ious = [float(m["q0"]["q0_best_iou"]) for m in matches if m["q0"].get("q0_best_iou") is not None]
    q0_reliable = [m for m in matches if float(m["q0"].get("q0_best_iou") or 0.0) >= 0.5]
    temporal_mapped = bool(q0_reliable)
    reliable_observation = any(m["event_row_reliable"] and float(m["q0"].get("q0_best_iou") or 0.0) >= 0.5 for m in matches)
    reasons: list[str] = []
    if not selected:
        reasons.append("event_track_rows_missing")
    if missing:
        reasons.append("q0_image_id_missing")
    if selected and not temporal_mapped:
        reasons.append("q0_temporal_iou_below_0.5")
    if temporal_mapped and not reliable_observation:
        reasons.append("event_row_not_reliable")
    if not reasons:
        reasons.append("mapped_and_event_row_reliable")
    return {
        "prefix": prefix,
        "selected_rows": len(selected),
        "q0_candidate_rows": sum(int(m["q0"].get("q0_candidate_count", 0)) for m in matches),
        "q0_rows_with_image": len(matches) - missing,
        "q0_temporal_mapped": temporal_mapped,
        "q0_reliable_iou_rows": len(q0_reliable),
        "q0_best_iou_max": max(q0_ious) if q0_ious else None,
        "q0_best_iou_mean": sum(q0_ious) / len(q0_ious) if q0_ious else None,
        "event_reliable_rows": reliable_event_rows,
        "reliable_observation": reliable_observation,
        "failure_reasons": reasons,
        "rows": matches,
    }


def align_event(event: Mapping[str, Any], by_track: Mapping[str, Sequence[Mapping[str, Any]]], q0_by_image: Mapping[int, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    source_keys = [str(x) for x in event.get("source_tracklet_keys", [])]
    source_rows: list[Mapping[str, Any]] = []
    for key in source_keys:
        source_rows.extend(by_track.get(key, ()))
    target_key = str(event.get("target_tracklet_key", ""))
    target_rows = list(by_track.get(target_key, ()))
    # Target row_keys are an explicit event-side ordering contract.  Prefer it
    # over the full track index and retain nulls for missing keys.
    target_by_key = {str(r.get("row_key", "")): r for r in target_rows}
    ordered_target = [target_by_key[k] for k in event.get("target_row_keys", []) if k in target_by_key]
    if ordered_target:
        target_rows = ordered_target
    records: list[dict[str, Any]] = []
    for prefix in PREFIXES:
        for role, rows in (("source", source_rows), ("target", target_rows)):
            role_audit = align_role(rows, q0_by_image, prefix)
            records.append({
                "event_key": str(event.get("event_key")),
                "fold": int(event.get("fold", -1)),
                "kind": str(event.get("kind", "")),
                "role": role,
                "prefix": prefix,
                "source_video": int(event.get("source_video")),
                "target_video": int(event.get("target_video")),
                "source_tracklet_keys": source_keys,
                "target_tracklet_key": target_key,
                "direct_track_key_intersection": False,
                "mapping_method": "evaluator_temporal_bbox_iou",
                "mapping_layer": "evaluator_only",
                "bbox_contract": "Q0 TAO xywh converted to xyxy; event CSV bbox_xyxy; same image_id and CSV dimensions",
                "assigned_semantics": "event CSV assigned only; Q0 TAO has no assigned field",
                "alignment": role_audit,
            })
    return records
