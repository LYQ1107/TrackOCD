"""Structured, evidence-carrying alignment failure taxonomy."""
from __future__ import annotations

FAILURE_CODES = ("NO_SHARED_NUMERIC_ID", "NO_DATASET_PROVENANCE", "NO_CANONICAL_VIDEO_KEY", "NO_CANONICAL_IMAGE_KEY",
 "ASSET_NOT_PRESENT_IN_EXISTING_Q0", "ASSET_FILE_MISSING", "PARTIAL_ASSET_MAPPING", "AMBIGUOUS_ASSET_MAPPING",
 "MISSING_EVENT_TRACKLET_ROWS", "MISSING_EVENT_FRAME_ID", "MISSING_EVENT_IMAGE_ID", "MISSING_EVENT_BBOX",
 "MISSING_Q0_FRAME_ID", "MISSING_Q0_PROPOSAL_LOCAL_ID", "Q0_IMAGE_PRESENT_NO_PREDICTION", "Q0_CANDIDATE_IOU_BELOW_0_5",
 "EVENT_ROW_NOT_RELIABLE", "MULTIPLE_ELIGIBLE_PHYSICAL_TRACKS", "PHYSICAL_TRACK_FRAGMENTATION", "CAUSAL_ORDER_VIOLATION",
 "Q0_REPLAY_INPUT_MISMATCH", "Q0_REPLAY_OUTPUT_MISMATCH", "TEXT_CATEGORY_DEPENDENCY", "INTERNAL_SCHEMA_ERROR")


def failure_record(event_key: str, role: str, tracklet_key: str, prefix: int, *, video_key: str | None, image_key: str | None,
                   source_file: str, source_ref: str | None, code: str, reason: str, available: dict, missing: list[str], recoverable: bool) -> dict:
    if code not in FAILURE_CODES: raise ValueError(code)
    return {"event_key": event_key, "role": role, "event_tracklet_key": tracklet_key, "prefix": int(prefix), "canonical_video_key": video_key,
            "canonical_image_key": image_key, "source_file": source_file, "source_ref": source_ref, "failure_code": code,
            "reason": reason, "available_evidence": available, "missing_evidence": missing, "recoverable": bool(recoverable)}
