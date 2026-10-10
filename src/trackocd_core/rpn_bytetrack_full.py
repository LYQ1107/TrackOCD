"""Complete-video current-only replay of one fixed existing classical tracker."""
import numpy as np
from src.trackocd_core.rpn_bytetrack import new_tracker, step, detector_digest


def replay_video(data, video, parameters, guard):
    before = detector_digest(data)
    tracker = new_tracker(parameters)
    arrays = {**data, "frame_offsets": [0], "track_id": [], "boxes": [], "score": []}
    statistics = {"frames": [], "empty_output_frames": 0}
    for ordinal, image in enumerate(video["images"]):
        guard()
        begin, end = map(int, data["det_offsets"][ordinal:ordinal + 2])
        output = step(tracker, data["det_boxes"][begin:end], data["det_score"][begin:end])
        arrays["boxes"].extend(output[:, :4].tolist())
        arrays["track_id"].extend(output[:, 4].astype(np.int64).tolist())
        arrays["score"].extend(output[:, 5].tolist())
        arrays["frame_offsets"].append(len(arrays["track_id"]))
        statistics["frames"].append({"ordinal": ordinal, "input_detections": end - begin, "output_tracks": len(output)})
        statistics["empty_output_frames"] += len(output) == 0
    arrays["boxes"] = np.asarray(arrays["boxes"], dtype=np.float32).reshape(-1, 4)
    arrays["score"] = np.asarray(arrays["score"], dtype=np.float32)
    arrays["track_id"] = np.asarray(arrays["track_id"], dtype=np.int64)
    arrays["frame_offsets"] = np.asarray(arrays["frame_offsets"], dtype=np.int64)
    if detector_digest(data) != before or detector_digest(arrays) != before:
        raise ValueError("No transformations of frozen raw detector arrays")
    return arrays, statistics
