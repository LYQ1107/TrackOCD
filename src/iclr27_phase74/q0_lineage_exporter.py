"""Conservative Q0 sidecar exporter.

The historical TAO JSON does not contain frame_id or proposal_local_id.  This
module emits explicit nulls and provenance rather than inventing either field.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Iterator, Mapping


def parse_track_key(video_id: Any, track_id: Any) -> str:
    return f"v{int(video_id)}:p{int(track_id)}"


def export_sidecar(q0_rows: Iterable[dict[str, Any]], *, checkpoint_sha256: str, config_sha256: str, code_commit: str,
                   physical_stream: str = "q0_existing") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions: dict[str, int] = defaultdict(int); rows: list[dict[str, Any]] = []; tracks: dict[str, dict[str, Any]] = {}
    for q in q0_rows:
        try:
            video, image, track = int(q["video_id"]), int(q["image_id"]), int(q["track_id"]); bbox = [float(x) for x in q.get("bbox", [])]; score = float(q.get("score", 0.0))
        except (KeyError, TypeError, ValueError): continue
        key = parse_track_key(video, track); pos = positions[key]; positions[key] += 1
        rec = {"schema_version": "phase74.q0_physical_lineage.v1", "physical_stream": physical_stream, "dataset_split": "validation",
               "canonical_video_key": None, "canonical_image_key": None, "video_id": video, "frame_id": None, "image_id": image,
               "proposal_local_id": None, "physical_track_id": track, "physical_row_key": None, "bbox_xywh": bbox, "bbox_xyxy": [bbox[0], bbox[1], bbox[0] + max(0., bbox[2]), bbox[1] + max(0., bbox[3])] if len(bbox) == 4 else None,
               "base_score": score, "candidate_rank_pre_filter": None, "candidate_rank_post_filter": None, "parent_physical_track_id": None,
               "lifecycle_state": "UNKNOWN", "source_checkpoint_sha256": checkpoint_sha256, "source_config_sha256": config_sha256, "source_code_commit": code_commit,
               "category_field_present_in_raw_output": "category_id" in q, "category_used_as_model_input": False, "event_metadata_used_as_model_input": False,
               "lineage_status": "UNRECOVERABLE_FROM_TAO_ONLY", "track_position_in_export": pos}
        rows.append(rec); t = tracks.setdefault(key, {"physical_stream": physical_stream, "canonical_video_key": None, "physical_track_id": track, "video_id": video, "row_count": 0, "image_ids": [], "frame_ids": [], "lineage_status": "UNRECOVERABLE_FROM_TAO_ONLY"}); t["row_count"] += 1; t["image_ids"].append(image)
    for t in tracks.values():
        t["first_image_id"] = min(t["image_ids"]) if t["image_ids"] else None; t["last_image_id"] = max(t["image_ids"]) if t["image_ids"] else None; t["frame_ids_known"] = False
    return rows, list(tracks.values())


def iter_sidecar(q0_rows: Iterable[dict[str, Any]], *, checkpoint_sha256: str, config_sha256: str, code_commit: str,
                 asset_by_image: Mapping[int, Mapping[str, Any]] | None = None,
                 physical_stream: str = "q0_existing") -> Iterator[dict[str, Any]]:
    """Memory-bounded variant used for the 1.2M-row frozen stream."""
    positions: dict[str, int] = defaultdict(int)
    for q in q0_rows:
        try:
            video, image, track = int(q["video_id"]), int(q["image_id"]), int(q["track_id"])
            bbox = [float(x) for x in q.get("bbox", [])]; score = float(q.get("score", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        key = parse_track_key(video, track); pos = positions[key]; positions[key] += 1; asset = (asset_by_image or {}).get(image, {})
        xyxy = [bbox[0], bbox[1], bbox[0] + max(0., bbox[2]), bbox[1] + max(0., bbox[3])] if len(bbox) == 4 else None
        yield {"schema_version": "phase74.q0_physical_lineage.v1", "physical_stream": physical_stream, "dataset_split": "validation",
               "canonical_video_key": asset.get("canonical_video_key"), "canonical_image_key": asset.get("canonical_image_key"), "video_id": video,
               "frame_id": None, "image_id": image, "proposal_local_id": None, "physical_track_id": track, "physical_row_key": None,
               "bbox_xywh": bbox, "bbox_xyxy": xyxy, "base_score": score, "candidate_rank_pre_filter": None, "candidate_rank_post_filter": None,
               "parent_physical_track_id": None, "lifecycle_state": "UNKNOWN", "source_checkpoint_sha256": checkpoint_sha256,
               "source_config_sha256": config_sha256, "source_code_commit": code_commit, "category_field_present_in_raw_output": "category_id" in q,
               "category_used_as_model_input": False, "event_metadata_used_as_model_input": False, "lineage_status": "UNRECOVERABLE_FROM_TAO_ONLY",
               "track_position_in_export": pos}


def five_field_key(video_id: Any, frame_id: Any, proposal_local_id: Any, track_id: Any, image_id: Any) -> str:
    return ":".join(str(x) for x in (video_id, frame_id, proposal_local_id, track_id, image_id))


def parse_five_field_key(key: str) -> tuple[str, str, str, str, str]:
    parts = str(key).split(":")
    if len(parts) != 5: raise ValueError(key)
    return tuple(parts)  # type: ignore[return-value]
