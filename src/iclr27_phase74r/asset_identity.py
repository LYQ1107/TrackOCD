"""Split-aware, duplicate-preserving asset identity for Phase74R."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from .io import canonical_hash, sha256


def normalize_path(value: Any) -> str:
    return re.sub(r"/+", "/", str(value or "").replace("\\", "/")).lstrip("./")


def remove_split_prefix(value: str) -> str:
    value = normalize_path(value)
    parts = value.split("/")
    if parts and parts[0].lower() in {"train", "val", "validation", "test", "dev"}:
        return "/".join(parts[1:])
    return value


def protocol_asset_key(record: dict[str, Any]) -> str:
    return str(record.get("canonical_image_key") or "")


def content_asset_key(record: dict[str, Any]) -> str:
    dataset = str(record.get("dataset_name", "unknown")).lower()
    video = remove_split_prefix(str(record.get("video_file_name", "")))
    frame = record.get("frame_index")
    try:
        frame_part = f"frame={int(frame)}"
    except (TypeError, ValueError):
        frame_part = f"file={remove_split_prefix(str(record.get('image_file_name', '')))}"
    return f"{dataset}|{video}|{frame_part}"


def file_identity_key(path: str | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    return sha256(Path(path))


@dataclass(frozen=True)
class AssetRecord:
    namespace: str
    dataset_name: str
    dataset_split: str
    video_id: int | None
    image_id: int | None
    video_file_name: str
    image_file_name: str
    frame_index: int | None
    resolved_path: str | None
    path_exists: bool
    protocol_asset_key: str
    content_asset_key: str
    file_identity_key: str | None = None

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def record_from_manifest(raw: dict[str, Any], namespace: str) -> AssetRecord:
    return AssetRecord(
        namespace=namespace,
        dataset_name=str(raw.get("dataset_name", "unknown")),
        dataset_split=str(raw.get("dataset_split", "unknown")),
        video_id=_int_or_none(raw.get("video_id", raw.get("event_video_id"))),
        image_id=_int_or_none(raw.get("image_id", raw.get("event_image_id"))),
        video_file_name=str(raw.get("video_file_name", "")),
        image_file_name=str(raw.get("image_file_name", "")),
        frame_index=_int_or_none(raw.get("frame_index")),
        resolved_path=str(raw.get("resolved_path")) if raw.get("resolved_path") else None,
        path_exists=bool(raw.get("path_exists", False)),
        protocol_asset_key=protocol_asset_key(raw),
        content_asset_key=content_asset_key(raw),
    )


def _safe_hash(path: str | None, cache: dict[str, str | None]) -> str | None:
    if not path or not Path(path).is_file():
        return None
    if path not in cache:
        cache[path] = file_identity_key(path)
    return cache[path]


def build_identity_records(q0_records: Iterable[dict[str, Any]], event_records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    q0 = [record_from_manifest(row, "q0_validation") for row in q0_records]
    events = [record_from_manifest(row, "event_train") for row in event_records]
    q0_by_content: dict[str, list[AssetRecord]] = {}
    event_by_content: dict[str, list[AssetRecord]] = {}
    for record in q0:
        q0_by_content.setdefault(record.content_asset_key, []).append(record)
    for record in events:
        event_by_content.setdefault(record.content_asset_key, []).append(record)
    hash_cache: dict[str, str | None] = {}
    mapping: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []
    event_status: list[dict[str, Any]] = []
    for event in events:
        candidates = q0_by_content.get(event.content_asset_key, [])
        if not candidates:
            status = "NO_CONTENT_MATCH"
            unresolved.append({"event": event.as_record(), "status": status, "candidate_count": 0})
        elif len(candidates) > 1:
            status = "MULTIPLE_CONTENT_MATCH"
            ambiguities.append({"event": event.as_record(), "status": status, "candidates": [x.as_record() for x in candidates]})
        else:
            candidate = candidates[0]
            event_hash = _safe_hash(event.resolved_path, hash_cache)
            q0_hash = _safe_hash(candidate.resolved_path, hash_cache)
            if event_hash is not None and q0_hash is not None and event_hash == q0_hash:
                status = "FILE_HASH_CONFIRMED"
            elif event.protocol_asset_key == candidate.protocol_asset_key:
                status = "ANNOTATION_ALIAS"
            elif event_hash is not None and q0_hash is not None and event_hash != q0_hash:
                status = "FILE_HASH_CONFLICT"
                ambiguities.append({"event": event.as_record(), "status": status, "candidate": candidate.as_record(), "event_file_hash": event_hash, "q0_file_hash": q0_hash})
            else:
                status = "UNIQUE_CONTENT_MATCH"
            if status not in {"FILE_HASH_CONFLICT"}:
                mapping.append({
                    "content_asset_key": event.content_asset_key,
                    "event_protocol_asset_key": event.protocol_asset_key,
                    "q0_protocol_asset_key": candidate.protocol_asset_key,
                    "event_image_id": event.image_id,
                    "q0_image_id": candidate.image_id,
                    "status": status,
                    "one_to_one": True,
                    "category_used": False,
                    "track_id_used": False,
                    "bbox_used": False,
                    "event_file_identity_key": event_hash,
                    "q0_file_identity_key": q0_hash,
                })
        event_status.append({
            "event_protocol_asset_key": event.protocol_asset_key,
            "content_asset_key": event.content_asset_key,
            "status": status,
            "q0_candidate_count": len(candidates),
            "event_path_exists": event.path_exists,
        })
    status_counts: dict[str, int] = {}
    for row in event_status:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    return {
        "q0_records": q0,
        "event_records": events,
        "q0_by_content": q0_by_content,
        "event_by_content": event_by_content,
        "mapping": mapping,
        "unresolved": unresolved,
        "ambiguities": ambiguities,
        "event_status": event_status,
        "summary": {
            "q0_records": len(q0),
            "event_records": len(events),
            "q0_content_keys": len(q0_by_content),
            "event_content_keys": len(event_by_content),
            "content_key_intersection": len(set(q0_by_content) & set(event_by_content)),
            "mapped_event_records": len(mapping),
            "unresolved_event_records": len(unresolved),
            "ambiguous_or_conflict_records": len(ambiguities),
            "duplicate_q0_content_keys": sum(len(v) > 1 for v in q0_by_content.values()),
            "duplicate_event_content_keys": sum(len(v) > 1 for v in event_by_content.values()),
            "status_counts": status_counts,
            "identity_does_not_use_category_or_track_id": True,
            "file_hashes_computed_only_for_unique_or_conflicting_candidates": True,
        },
    }


def synthetic_pipeline(record: dict[str, Any], q0_record: dict[str, Any]) -> dict[str, Any]:
    """A tiny physical-side pipeline used by metamorphic contract tests."""
    event = record_from_manifest(record, "event_train")
    q0 = record_from_manifest(q0_record, "q0_validation")
    return {
        "content_asset_key": event.content_asset_key,
        "mapped": event.content_asset_key == q0.content_asset_key,
        "bbox_iou": float(record.get("bbox_iou", q0_record.get("bbox_iou", 0.0))),
        "joint_reliable": bool(record.get("assigned", 0) == 1 and float(record.get("event_iou", 0.0)) >= 0.5 and float(record.get("q0_iou", 0.0)) >= 0.5),
        "fragmentation_signature": list(record.get("fragmentation_signature", [])),
    }
