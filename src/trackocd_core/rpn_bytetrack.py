"""Current-frame-only fixed ByteTrack on already frozen generic RPN boxes.

No detector rerun, semantic/prototype/category/native-track input, score repair
or new learned weights. Source/config identity is checked before replay.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.physical_qualification import json_digest
from src.trackocd_core.amg_physical_diagnostic import validate_arrays

PARAMETERS = {"track_thresh": .5, "low_thresh": .1, "match_thresh": .8, "new_track_thresh": .6,
              "track_buffer": 30, "frame_rate": 30, "min_box_area": 10, "mot20": False, "filter_mot_aspect": False}


def read_detector_video(path, video):
    fields = ("image_id", "frame_index", "det_offsets", "det_boxes", "det_score")
    with np.load(path, allow_pickle=False) as arrays:
        data = {key: arrays[key].copy() for key in fields}  # Never access native IDs/track boxes.
    for key in ("image_id", "frame_index"):
        if data[key].dtype.kind not in "iu" or data[key].tolist() != [i[key] for i in video["images"]]:
            raise ValueError("Wrong detector chronology/universe; no frame dropping")
    offsets, boxes, scores = data["det_offsets"], data["det_boxes"], data["det_score"]
    if (offsets.dtype.kind not in "iu" or offsets.shape != (len(video["images"]) + 1,) or offsets[0] != 0
            or np.any(np.diff(offsets.astype(np.int64)) < 0) or offsets[-1] != len(scores) or np.any(offsets > len(scores))
            or boxes.shape != (len(scores), 4) or scores.shape != (len(scores),)
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Invalid raw RPN detections; no repair/calibration")
    return data


def detector_digest(data):
    digest = hashlib.sha256()
    for key in ("image_id", "frame_index", "det_offsets", "det_boxes", "det_score"):
        array = data[key]
        digest.update(key.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def new_tracker(parameters):
    if parameters != PARAMETERS:
        raise ValueError("Use exact frozen prior ByteTrack parameters; no search")
    from src.iclr27_phase3b.bytetrack import BYTETracker
    tracker = BYTETracker(**parameters)
    tracker.reset()  # Explicit per-video physical-ID reset; no semantic state.
    return tracker


def step(tracker, boxes_xyxy, foreground_scores):
    boxes, scores = np.asarray(boxes_xyxy), np.asarray(foreground_scores)
    if (boxes.shape != (len(scores), 4) or scores.ndim != 1 or not np.isfinite(boxes).all()
            or not np.isfinite(scores).all() or np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Current anonymous boxes/probabilities only; no repair")
    output = tracker.update(np.column_stack((boxes, scores)))
    if (output.shape != (len(output), 6) or not np.isfinite(output).all() or np.any(output[:, 2:4] <= output[:, :2])
            or np.any(output[:, 4] < 0) or np.any(output[:, 4] != np.floor(output[:, 4]))
            or len(np.unique(output[:, 4])) != len(output) or not np.isin(output[:, 5], scores).all()):
        raise ValueError("Invalid classical output/score changed; preserve and stop")
    return output.copy()


def output_digest(rows):
    return hashlib.sha256(str(rows.dtype).encode() + str(rows.shape).encode() + rows.tobytes()).hexdigest()


def causal_probe(data, parameters):
    if len(data["image_id"]) < 4:
        raise ValueError("Four fixed first frames required; no alternative selection")
    replays = []
    for changed in (False, True):
        tracker, frames = new_tracker(parameters), []
        for ordinal in range(4):
            begin, end = map(int, data["det_offsets"][ordinal:ordinal + 2])
            boxes, scores = data["det_boxes"][begin:end].copy(), data["det_score"][begin:end].copy()
            if changed and ordinal >= 2:
                boxes += np.asarray([1000, 1000, 1000, 1000], dtype=boxes.dtype)
            rows = step(tracker, boxes, scores)
            frames.append({"input_sha256": hashlib.sha256(boxes.tobytes() + scores.tobytes()).hexdigest(),
                           "output_sha256": output_digest(rows), "tracks": len(rows)})
        replays.append(frames)
    first, changed = replays
    valid = (all(first[i]["input_sha256"] == changed[i]["input_sha256"] and first[i]["output_sha256"] == changed[i]["output_sha256"] for i in (0, 1))
             and all(first[i]["input_sha256"] != changed[i]["input_sha256"] for i in (2, 3))
             and all(first[i]["tracks"] > 0 for i in (0, 1)))
    if not valid:
        raise ValueError("Nontrivial exact prefix/future-change check failed")
    return {"pass": True, "frame_updates": 8, "replays": replays,
            "scope": "First4 real cached frames, exact first2 outputs; future boxes only shifted, no score repair or production changes"}


def seal_video(run, video, arrays, config_sha256, source_sha256, tracker_source_sha256, statistics):
    validate_arrays(arrays, video)
    directory = run / "shards"; filename = f"video_{video['video_id']:04d}.npz"
    path, marker = directory / filename, directory / f"video_{video['video_id']:04d}.complete.json"
    if path.exists() or marker.exists():
        raise ValueError("Never overwrite completed or partial classical shard")
    temporary = path.with_name("." + filename + ".tmp")
    with temporary.open("xb") as writer:
        np.savez_compressed(writer, **arrays); writer.flush(); os.fsync(writer.fileno())
    os.replace(temporary, path)
    row = {"status": "COMPLETE_CLASSICAL_VIDEO", "video_id": video["video_id"], "images": len(video["images"]),
           "config_sha256": config_sha256, "video_plan_sha256": json_digest(video), "source_npz_sha256": source_sha256,
           "tracker_source_sha256": tracker_source_sha256, "npz_filename": filename, "npz_bytes": path.stat().st_size,
           "npz_sha256": sha256_file(path), "prediction_rows": len(arrays["track_id"]), "detection_rows": len(arrays["det_score"]),
           "source_detector_arrays_sha256": detector_digest(arrays), "frame_statistics": statistics,
           "learned_model_weights": False, "nn_frozen_state_claim": False}
    atomic_json(marker, row)
    return row


def completed_video(run, video, config_sha256, tracker_source_sha256):
    marker = run / "shards" / f"video_{video['video_id']:04d}.complete.json"
    if not marker.exists():
        return None
    if marker.is_symlink(): raise ValueError("Unexpected classical marker link")
    row = json.loads(marker.read_text()); filename = Path(row["npz_filename"]); path = marker.parent / filename
    if (row["status"] != "COMPLETE_CLASSICAL_VIDEO" or row["video_id"] != video["video_id"] or row["images"] != len(video["images"])
            or row["config_sha256"] != config_sha256 or row["tracker_source_sha256"] != tracker_source_sha256
            or row["video_plan_sha256"] != json_digest(video) or filename.name != str(filename) or path.is_symlink()
            or row["npz_bytes"] != path.stat().st_size or row["npz_sha256"] != sha256_file(path)):
        raise ValueError("Changed classical complete shard; no rerun")
    with np.load(path, allow_pickle=False) as arrays:
        validate_arrays(arrays, video)
        if (len(arrays["track_id"]) != row["prediction_rows"] or len(arrays["det_score"]) != row["detection_rows"]
                or detector_digest(arrays) != row["source_detector_arrays_sha256"]):
            raise ValueError("Changed output counts/detector lineage")
    return row
