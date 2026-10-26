#!/usr/bin/env python3
"""Causally re-associate sparse anonymous tracklets with box velocity.

This diagnostic consumes only an existing anonymous ``train_predicted``
physical stream.  A birth may join a prior persistent state using a fixed
source-frame horizon and IoU against a constant-velocity extrapolation from
that state's last two observations.  The extrapolation is prefix-only: future
observations never enter the state used for a current decision.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from scipy.optimize import linear_sum_assignment

from scripts.trackocd_cvpr.causal_physical_reassociate import box_iou


def _predicted_box(state: dict[str, Any], frame: int) -> tuple[list[float], str]:
    last_frame = int(state["last_frame"])
    last_box = [float(value) for value in state["last_box"]]
    previous_frame = state.get("previous_frame")
    previous_box = state.get("previous_box")
    if previous_frame is None or previous_box is None or last_frame <= int(previous_frame):
        return last_box, "last_box_fallback"
    denominator = last_frame - int(previous_frame)
    scale = (int(frame) - last_frame) / denominator
    velocity = [
        (last_box[index] - float(previous_box[index])) / denominator
        for index in range(4)
    ]
    return [last_box[index] + velocity[index] * scale * denominator for index in range(4)], "constant_velocity"


def velocity_reassociate_video(
    rows: list[dict[str, Any]],
    *,
    iou_threshold: float,
    max_gap_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a causal velocity-reassociated stream and compact counts."""

    events_by_frame: dict[int, list[tuple[str, int, int, list[float], str, float]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        native_id = str(row["physical_track_id"])
        for position, (frame, box, image_path, quality) in enumerate(
            zip(row["frame_ids"], row["boxes_xyxy"], row["image_paths"], row["quality"])
        ):
            events_by_frame[int(frame)].append(
                (native_id, row_index, position, [float(value) for value in box], str(image_path), float(quality))
            )

    native_to_persistent: dict[str, int] = {}
    persistent_state: dict[int, dict[str, Any]] = {}
    output: dict[int, dict[str, Any]] = {}
    next_persistent = 0
    matched_births = 0
    velocity_matches = 0
    fallback_matches = 0
    new_births = 0
    collision_splits = 0
    observations = 0

    for frame in sorted(events_by_frame):
        frame_events = sorted(events_by_frame[frame], key=lambda item: (item[0], item[1], item[2]))
        assigned: list[tuple[tuple[str, int, int, list[float], str, float], int]] = []
        births: list[tuple[str, int, int, list[float], str, float]] = []
        used_persistent: set[int] = set()

        for event in frame_events:
            native_id = event[0]
            persistent_id = native_to_persistent.get(native_id)
            if persistent_id is not None and persistent_id not in used_persistent:
                state = persistent_state[persistent_id]
                if state["last_frame"] < frame:
                    assigned.append((event, persistent_id))
                    used_persistent.add(persistent_id)
                    continue
                collision_splits += 1
            births.append(event)

        candidate_ids = [
            persistent_id
            for persistent_id, state in sorted(persistent_state.items())
            if persistent_id not in used_persistent
            and state["last_frame"] < frame
            and frame - state["last_frame"] <= max_gap_frames
        ]
        if births and candidate_ids:
            predictions = {
                persistent_id: _predicted_box(persistent_state[persistent_id], frame)
                for persistent_id in candidate_ids
            }
            matrix = [
                [box_iou(event[3], predictions[persistent_id][0]) for persistent_id in candidate_ids]
                for event in births
            ]
            import numpy as np

            weights = np.asarray(matrix, dtype=float)
            rows_index, columns_index = linear_sum_assignment(weights, maximize=True)
            matched_birth_indexes: set[int] = set()
            for birth_index, candidate_index in zip(rows_index, columns_index):
                if weights[int(birth_index), int(candidate_index)] < iou_threshold:
                    continue
                event = births[int(birth_index)]
                persistent_id = candidate_ids[int(candidate_index)]
                assigned.append((event, persistent_id))
                used_persistent.add(persistent_id)
                native_to_persistent[event[0]] = persistent_id
                matched_birth_indexes.add(int(birth_index))
                matched_births += 1
                if predictions[persistent_id][1] == "constant_velocity":
                    velocity_matches += 1
                else:
                    fallback_matches += 1
            births = [event for index, event in enumerate(births) if index not in matched_birth_indexes]

        for event in births:
            persistent_id = next_persistent
            next_persistent += 1
            new_births += 1
            native_to_persistent[event[0]] = persistent_id
            assigned.append((event, persistent_id))
            used_persistent.add(persistent_id)

        for event, persistent_id in assigned:
            native_id, row_index, position, box, image_path, quality = event
            if persistent_id not in output:
                source_row = rows[row_index]
                output[persistent_id] = {
                    "video_id": int(source_row["video_id"]),
                    "frame_ids": [],
                    "boxes_xyxy": [],
                    "image_paths": [],
                    "quality": [],
                }
            if persistent_id in persistent_state and persistent_state[persistent_id]["last_frame"] == frame:
                raise ValueError("causal velocity reassociation produced a same-frame duplicate")
            prior = persistent_state.get(persistent_id)
            persistent_state[persistent_id] = {
                "previous_frame": None if prior is None else prior["last_frame"],
                "previous_box": None if prior is None else prior["last_box"],
                "last_frame": frame,
                "last_box": box,
                "owner_native_id": native_id,
            }
            output[persistent_id]["frame_ids"].append(frame)
            output[persistent_id]["boxes_xyxy"].append(box)
            output[persistent_id]["image_paths"].append(image_path)
            output[persistent_id]["quality"].append(quality)
            observations += 1

    result_rows: list[dict[str, Any]] = []
    for stream_order, persistent_id in enumerate(sorted(output)):
        row = output[persistent_id]
        video_id = row["video_id"]
        result_rows.append({
            "sample_key": f"causal_velocity_reassoc_{video_id}_{persistent_id}",
            "source_sample_id": f"causal_velocity_reassoc_{video_id}_{persistent_id}",
            "video_id": video_id,
            "physical_track_id": f"causal-velocity-reassoc-{persistent_id:08d}",
            "frame_ids": row["frame_ids"],
            "boxes_xyxy": row["boxes_xyxy"],
            "image_paths": row["image_paths"],
            "quality": row["quality"],
            "stream_order": stream_order,
            "source_split": "train_predicted",
        })
    return result_rows, {
        "input_native_tracks": len(rows),
        "output_persistent_tracks": len(result_rows),
        "input_observations": sum(len(row["frame_ids"]) for row in rows),
        "output_observations": observations,
        "matched_births": matched_births,
        "velocity_matches": velocity_matches,
        "fallback_matches": fallback_matches,
        "new_births": new_births,
        "collision_splits": collision_splits,
    }
