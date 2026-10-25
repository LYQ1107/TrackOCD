#!/usr/bin/env python3
"""Causally re-associate category-free physical tracklets.

This diagnostic consumes only an existing anonymous ``train_predicted`` physical
stream.  It processes observations in source-frame order and may join a new
native track ID only to a prior-frame persistent state using a fixed one-frame
IoU bridge.  It never opens GT, labels, vocabulary, pixels, or feature arrays.
The native-to-derived mapping is transient and is not written to the receipt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/trackocd_cvpr/causal_physical_reassociation_smoke_v1.json"
PHYSICAL_FIELDS = {
    "sample_key",
    "source_sample_id",
    "video_id",
    "physical_track_id",
    "frame_ids",
    "boxes_xyxy",
    "image_paths",
    "quality",
    "stream_order",
    "source_split",
}
FORBIDDEN_FIELDS = {
    "category_id",
    "category_ids",
    "category_name",
    "gt_category_id",
    "gt_split",
    "gt_match_id",
    "text",
    "novel",
    "label",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_box(box: Any) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in box)
        and float(box[2]) > float(box[0])
        and float(box[3]) > float(box[1])
    )


def box_iou(left: list[float], right: list[float]) -> float:
    ix0 = max(float(left[0]), float(right[0]))
    iy0 = max(float(left[1]), float(right[1]))
    ix1 = min(float(left[2]), float(right[2]))
    iy1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def load_rows(path: Path, video_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    rows_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, str]] = set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict) or set(row) != PHYSICAL_FIELDS:
                raise ValueError(f"physical stream schema mismatch at {path}:{line_number}")
            if FORBIDDEN_FIELDS & set(row) or row.get("source_split") != "train_predicted":
                raise ValueError(f"forbidden or non-Train row at {path}:{line_number}")
            video_id = int(row["video_id"])
            if video_id not in video_ids:
                continue
            track_id = str(row["physical_track_id"])
            key = (video_id, track_id)
            if key in seen:
                raise ValueError(f"duplicate native track {key}")
            seen.add(key)
            frames = row["frame_ids"]
            boxes = row["boxes_xyxy"]
            paths = row["image_paths"]
            quality = row["quality"]
            if (
                not isinstance(frames, list)
                or not isinstance(boxes, list)
                or not isinstance(paths, list)
                or not isinstance(quality, list)
                or not frames
                or len(frames) != len(boxes)
                or len(frames) != len(paths)
                or len(frames) != len(quality)
                or len(set(frames)) != len(frames)
            ):
                raise ValueError(f"invalid native lineage at {path}:{line_number}")
            for box in boxes:
                if not finite_box(box):
                    raise ValueError(f"invalid box at {path}:{line_number}")
            rows_by_video[video_id].append(row)
    if set(rows_by_video) != video_ids:
        raise ValueError(f"stream videos {sorted(rows_by_video)} != expected {sorted(video_ids)}")
    return rows_by_video


def reassociate_video(
    rows: list[dict[str, Any]],
    *,
    iou_threshold: float,
    max_gap_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
            matrix = [
                [box_iou(event[3], persistent_state[persistent_id]["last_box"]) for persistent_id in candidate_ids]
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
                raise ValueError("causal re-association produced a same-frame duplicate")
            persistent_state[persistent_id] = {
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
        derived_id = f"causal-reassoc-{persistent_id:08d}"
        result_rows.append({
            "sample_key": f"causal_reassoc_{video_id}_{persistent_id}",
            "source_sample_id": f"causal_reassoc_{video_id}_{persistent_id}",
            "video_id": video_id,
            "physical_track_id": derived_id,
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
        "new_births": new_births,
        "collision_splits": collision_splits,
    }


def run(mode: str) -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("status") != "FROZEN_CATEGORY_FREE_CAUSAL_REASSOCIATION_DIAGNOSTIC":
        raise ValueError("unexpected reassociation protocol status")
    if mode not in cfg["fixed_subsets"]:
        raise ValueError(f"unknown fixed subset {mode}")
    subset = cfg["fixed_subsets"][mode]
    video_ids = {int(value) for value in subset["video_ids"]}
    source = ROOT / subset["source_stream"]
    output = ROOT / subset["derived_stream"]
    receipt = ROOT / subset["receipt"]
    if output.exists() or receipt.exists():
        raise RuntimeError("refusing to overwrite existing reassociation output")
    started = time.monotonic()
    rows_by_video = load_rows(source, video_ids)
    all_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for video_id in sorted(video_ids):
        rows, summary = reassociate_video(
            rows_by_video[video_id],
            iou_threshold=0.5,
            max_gap_frames=1,
        )
        all_rows.extend(rows)
        for key, value in summary.items():
            counts[key] += int(value)
    for stream_order, row in enumerate(all_rows):
        row["stream_order"] = stream_order
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{__import__('os').getpid()}")
    if temporary.exists():
        raise RuntimeError("temporary output already exists")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in all_rows:
            encoded = (json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
    temporary.replace(output)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "trackocd.cvpr.causal-physical-reassociation-result.v1",
        "status": "COMPLETE_CATEGORY_FREE_CAUSAL_REASSOCIATION_DIAGNOSTIC",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256_file(CONFIG),
        "mode": mode,
        "video_ids": sorted(video_ids),
        "source_stream": {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "derived_stream": {
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": digest.hexdigest(),
        },
        "algorithm": {
            "iou_threshold": 0.5,
            "max_gap_frames": 1,
            "native_to_derived_mapping_persisted": False,
            "future_observations_used": False,
            "semantic_or_feature_input": False,
        },
        "counts": dict(counts),
        "source_binding": {
            "kind": "derived_category_free_causal_physical_reassociation",
            "algorithm": "one-frame prior-state Hungarian IoU bridge",
            "input_stream_sha256": sha256_file(source),
            "config_sha256": sha256_file(CONFIG),
        },
        "boundaries": {
            "train_annotation_opened": False,
            "gt_category_fields_read": False,
            "gt_role_or_vocabulary_read": False,
            "val_labels_read": False,
            "test_accessed": False,
            "feature_arrays_read": False,
            "training_started": False,
            "formal_mot_or_ocd_metrics": False,
            "model_selection": False,
            "raw_payload_uploaded": False,
        },
        "resources": {
            "device": "CPU",
            "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        },
        "interpretation": "Mechanism evidence only; no official OCD/MOT metric, frontend qualification, model selection, or Goal A/B/C acceptance.",
    }
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["policy_train_pseudo", "development_pseudo"], required=True)
    args = parser.parse_args()
    result = run(args.mode)
    print(json.dumps({"status": result["status"], "mode": result["mode"], "counts": result["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
