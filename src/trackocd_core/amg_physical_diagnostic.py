"""SAM-grid video-atomic seal: full mask-derived output, no inherited cap50."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.physical_qualification import json_digest


def validate_arrays(arrays, video):
    images = video["images"]
    for field in ("image_id", "frame_index"):
        values = np.asarray(arrays[field])
        if values.dtype.kind not in "iu" or values.tolist() != [i[field] for i in images]:
            raise ValueError("AMG shard missing/changed registered frame metadata")
    for prefix, bb, ss, ii in (("frame", "boxes", "score", "track_id"), ("det", "det_boxes", "det_score", None)):
        offsets, boxes, scores = np.asarray(arrays[prefix + "_offsets"]), np.asarray(arrays[bb]), np.asarray(arrays[ss])
        if (offsets.dtype.kind not in "iu" or offsets.shape != (len(images) + 1,) or offsets[0] != 0
                or np.any(offsets < 0) or np.any(np.diff(offsets.astype(np.int64)) < 0)
                or int(offsets[-1]) != len(scores) or np.any(offsets > len(scores))
                or boxes.shape != (len(scores), 4) or scores.shape != (len(scores),)
                or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
                or np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any((scores < 0) | (scores > 1))
                or np.any(np.diff(offsets.astype(np.int64)) > 3072)):
            raise ValueError("Invalid AMG arrays; no repair/cap or frame dropping")
        # 3072 is the mathematical 32x32x3 raw mask limit, never a selected cap.
        if ii:
            ids = np.asarray(arrays[ii])
            if ids.dtype.kind not in "iu" or ids.shape != scores.shape or np.any(ids < 0):
                raise ValueError("Invalid anonymous physical IDs")
            for begin, end in zip(offsets, offsets[1:]):
                if len(np.unique(ids[begin:end])) != end - begin:
                    raise ValueError("Duplicate physical ID in a frame")


def seal_video(run, video, arrays, config_sha256, frame_stats):
    validate_arrays(arrays, video)
    if frame_stats.get("frozen_state_unchanged") is not True:
        raise ValueError("Frozen state required before atomic seal")
    directory = run / "shards"
    filename = f"video_{video['video_id']:04d}.npz"
    path, marker = directory / filename, directory / f"video_{video['video_id']:04d}.complete.json"
    if path.exists() or marker.exists():
        raise ValueError("Never overwrite a completed or partial AMG video")
    temporary = path.with_name("." + filename + ".tmp")
    with temporary.open("xb") as writer:
        np.savez_compressed(writer, **arrays); writer.flush(); os.fsync(writer.fileno())
    os.replace(temporary, path)
    record = {"status": "COMPLETE_VIDEO", "video_id": video["video_id"], "images": len(video["images"]),
              "config_sha256": config_sha256, "video_plan_sha256": json_digest(video), "npz_filename": filename,
              "npz_bytes": path.stat().st_size, "npz_sha256": sha256_file(path),
              "prediction_rows": len(arrays["track_id"]), "detection_rows": len(arrays["det_score"]),
              "frame_statistics": frame_stats}
    atomic_json(marker, record)
    return record


def completed_video(run, video, config_sha256):
    marker = run / "shards" / f"video_{video['video_id']:04d}.complete.json"
    if not marker.exists():
        return None
    if marker.is_symlink():
        raise ValueError("Unexpected marker link")
    row = json.loads(marker.read_text())
    filename = Path(row["npz_filename"]); path = marker.parent / filename
    if (row["status"] != "COMPLETE_VIDEO" or row["video_id"] != video["video_id"]
            or row["images"] != len(video["images"]) or row["config_sha256"] != config_sha256
            or row["video_plan_sha256"] != json_digest(video) or filename.name != str(filename)
            or row["frame_statistics"]["frozen_state_unchanged"] is not True or path.is_symlink()
            or path.stat().st_size != row["npz_bytes"] or sha256_file(path) != row["npz_sha256"]):
        raise ValueError("Changed AMG complete shard; no re-extraction")
    with np.load(path, allow_pickle=False) as arrays:
        validate_arrays(arrays, video)
    return row
