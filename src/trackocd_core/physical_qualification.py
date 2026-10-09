"""Full image-metadata universe and video-atomic frozen physical predictions."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np

from src.trackocd_v2.io import atomic_json, sha256_file


def metadata_universe(annotation: dict) -> list[dict]:
    """All videos/images, never inspect annotation/category/track labels."""
    ids = [int(v["id"]) for v in annotation["videos"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate video metadata")
    grouped, seen = defaultdict(list), set()
    for image in annotation["images"]:
        image_id, video_id = int(image["id"]), int(image["video_id"])
        if image_id in seen or video_id not in ids:
            raise ValueError("Duplicate image or unregistered video")
        seen.add(image_id)
        grouped[video_id].append({"image_id": image_id, "frame_index": int(image["frame_index"]),
                                  "image_path": image["file_name"]})
    rows = []
    for video_id in sorted(ids):
        images = sorted(grouped[video_id], key=lambda r: (r["frame_index"], r["image_id"]))
        if not images or any(b["frame_index"] <= a["frame_index"] for a, b in zip(images, images[1:])):
            raise ValueError("Empty video or non-increasing chronology; do not drop/substitute")
        rows.append({"video_id": video_id, "images": images})
    return rows


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def val_image(root: Path, relative: str) -> Path:
    path = Path(relative)
    if (path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "val"
            or not (root / path).resolve().is_relative_to((root / "val").resolve())
            or not (root / path).is_file()):
        raise ValueError("Missing or escaping Val image; preserve full universe")
    return root / path


def validate_arrays(arrays, video: dict) -> None:
    """No repairs, filtering or role fields; require every registered image."""
    images = video["images"]
    for field, key in (("image_id", "image_id"), ("frame_index", "frame_index")):
        values = np.asarray(arrays[field])
        if values.dtype.kind not in "iu" or values.tolist() != [i[key] for i in images]:
            raise ValueError("Physical shard does not cover registered metadata")
    for prefix, boxes_key, scores_key, ids_key in (("frame", "boxes", "score", "track_id"), ("det", "det_boxes", "det_score", None)):
        offsets = np.asarray(arrays[prefix + "_offsets"])
        boxes, scores = np.asarray(arrays[boxes_key]), np.asarray(arrays[scores_key])
        if (offsets.dtype.kind not in "iu" or offsets.shape != (len(images) + 1,) or offsets[0] != 0
                or np.any(offsets > len(scores)) or np.any(np.diff(offsets.astype(np.int64)) < 0) or int(offsets[-1]) != len(scores)
                or boxes.shape != (len(scores), 4) or scores.shape != (len(scores),)
                or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
                or np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any((scores < 0) | (scores > 1))
                or np.any(np.diff(offsets.astype(np.int64)) > 50)):
            raise ValueError("Invalid frozen anonymous prediction; do not repair")
        if ids_key:
            ids = np.asarray(arrays[ids_key])
            if ids.dtype.kind not in "iu" or ids.shape != scores.shape or np.any(ids < 0):
                raise ValueError("Invalid physical IDs")
            for begin, end in zip(offsets, offsets[1:]):
                if len(np.unique(ids[begin:end])) != end - begin:
                    raise ValueError("Duplicate physical ID in a frame")


def completed_video(run: Path, video: dict, config_sha256: str) -> dict | None:
    marker = run / "shards" / f"video_{video['video_id']:04d}.complete.json"
    if not marker.exists():
        return None
    if marker.is_symlink():
        raise ValueError("Unexpected completion link")
    record = json.loads(marker.read_text())
    filename = Path(record["npz_filename"])
    path = marker.parent / filename
    if (record["status"] != "COMPLETE_VIDEO" or record["config_sha256"] != config_sha256
            or record["video_id"] != video["video_id"] or record["images"] != len(video["images"])
            or record["frame_statistics"]["frozen_state_unchanged"] is not True
            or record["video_plan_sha256"] != json_digest(video) or filename.name != str(filename)
            or path.is_symlink() or path.stat().st_size != record["npz_bytes"]
            or sha256_file(path) != record["npz_sha256"]):
        raise ValueError("Changed completed shard; preserve and stop, not re-extract")
    with np.load(path, allow_pickle=False) as arrays:
        validate_arrays(arrays, video)
    return record


def seal_video(run: Path, video: dict, arrays: dict, config_sha256: str, attempt: str, frame_stats: dict) -> dict:
    validate_arrays(arrays, video)
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", attempt) or frame_stats.get("frozen_state_unchanged") is not True:
        raise ValueError("Invalid owned attempt or unverified frozen state")
    directory = run / "shards"
    marker = directory / f"video_{video['video_id']:04d}.complete.json"
    path = directory / f"video_{video['video_id']:04d}.{attempt}.npz"
    if marker.exists() or path.exists():
        raise ValueError("Never overwrite a completed or retained attempted shard")
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("xb") as writer:
        np.savez_compressed(writer, **arrays)
        writer.flush(); os.fsync(writer.fileno())
    os.replace(temporary, path)
    record = {"status": "COMPLETE_VIDEO", "video_id": video["video_id"], "images": len(video["images"]),
              "config_sha256": config_sha256, "video_plan_sha256": json_digest(video),
              "npz_filename": path.name, "npz_bytes": path.stat().st_size, "npz_sha256": sha256_file(path),
              "prediction_rows": len(arrays["track_id"]), "detection_rows": len(arrays["det_score"]),
              "attempt": attempt, "frame_statistics": frame_stats}
    atomic_json(marker, record)
    return record


def aggregate_purity(rows: list[dict]) -> dict:
    additive = ["physical_tracks", "prediction_rows", "matched_rows", "unmatched_unknown_rows", "observed_multicategory_tracks",
                "observed_multiple_gt_identity_tracks", "tracks_with_no_gt_match", "tracks_with_unknown_observations",
                "entire_observed_track_matched_to_one_category"]
    result = {key: sum(r[key] for r in rows) for key in additive}
    majority = sum(r["observed_majority_category_fraction_over_matched_rows"] * r["matched_rows"]
                   for r in rows if r["matched_rows"])
    result["observed_majority_category_fraction_over_matched_rows"] = majority / result["matched_rows"] if result["matched_rows"] else None
    result["incomplete_annotation_boundary"] = "Unmatched observations are unknown; an observed single category is not a global purity certificate"
    return result
