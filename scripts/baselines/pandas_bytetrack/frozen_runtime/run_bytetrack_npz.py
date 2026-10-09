#!/usr/bin/env python3
"""Run category-agnostic ByteTrack on frozen per-video PANDAS NPZ files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--track-thresh", type=float, default=0.5)
    parser.add_argument("--low-thresh", type=float, default=0.1)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--score-key", default="foreground_scores",
                        help="NPZ score field passed to ByteTrack")
    parser.add_argument("--new-track-thresh", type=float, default=None)
    parser.add_argument("--filter-mot-aspect", action="store_true",
                        help="enable the legacy pedestrian/MOT aspect-ratio filter")
    parser.add_argument("--sanity-videos", type=int, default=20)
    args = parser.parse_args()

    trackocd_root = Path("/data3/liuyeqiang/TrackOCD")
    sys.path.insert(0, str(trackocd_root))
    from src.iclr27_phase3b.bytetrack import BYTETracker

    files = sorted(args.detections_root.glob("video_*.npz"))
    if not files:
        raise RuntimeError(f"no detections found in {args.detections_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    sanity = []
    total_frames = 0
    total_detections = 0
    total_track_rows = 0
    total_tracks = 0
    total_frames_with_detections = 0
    total_empty_frames = 0
    start = time.time()

    for video_index, detection_file in enumerate(files):
        video_id = int(detection_file.stem.split("_")[-1])
        with np.load(detection_file) as data:
            frame_index = data["frame_index"]
            image_id = data["image_id"]
            offsets = data["frame_offsets"]
            boxes = data["boxes"]
            if args.score_key not in data.files:
                raise KeyError(f"missing tracking score field {args.score_key} in {detection_file}")
            scores = data[args.score_key]
            if not np.isfinite(scores).all() or not ((0.0 <= scores).all() and (scores <= 1.0).all()):
                raise ValueError(f"tracking scores out of range in {detection_file}")
            semantic_scores = data["scores"] if "scores" in data.files else None
            prototypes = data["prototype_id"]
            out_frame_index = []
            out_image_id = []
            out_track_id = []
            out_boxes = []
            out_scores = []
            out_semantic_scores = []
            out_prototypes = []
            out_offsets = [0]
            tracker = BYTETracker(
                track_thresh=args.track_thresh,
                low_thresh=args.low_thresh,
                match_thresh=args.match_thresh,
                track_buffer=args.track_buffer,
                frame_rate=args.frame_rate,
                new_track_thresh=args.new_track_thresh,
                filter_mot_aspect=args.filter_mot_aspect,
            )
            observed_track_ids = set()
            high = 0
            low = 0
            frames_with_detections = 0
            for frame_pos in range(len(frame_index)):
                begin, end = int(offsets[frame_pos]), int(offsets[frame_pos + 1])
                frame_boxes = boxes[begin:end]
                frame_scores = scores[begin:end]
                frame_prototypes = prototypes[begin:end]
                if len(frame_scores):
                    frames_with_detections += 1
                    high += int(np.sum(frame_scores > args.track_thresh))
                    low += int(np.sum((frame_scores > args.low_thresh) & (frame_scores <= args.track_thresh)))
                bt_input = np.concatenate(
                    [frame_boxes, frame_scores[:, None]], axis=1
                ) if len(frame_boxes) else np.empty((0, 5), dtype=np.float32)
                tracked_rows = tracker.update(bt_input)
                for tracked in tracked_rows:
                    track_box = tracked[:4].astype(np.float32)
                    track_id = int(tracked[4])
                    score = float(tracked[5])
                    overlaps = iou_one(track_box, frame_boxes)
                    if len(overlaps):
                        detection_index = int(np.argmax(overlaps))
                        prototype_id = int(frame_prototypes[detection_index])
                        semantic_score = (
                            float(semantic_scores[begin + detection_index])
                            if semantic_scores is not None else float("nan")
                        )
                    else:
                        prototype_id = -1
                        semantic_score = float("nan")
                    out_track_id.append(track_id)
                    out_boxes.append(track_box)
                    out_scores.append(score)
                    out_semantic_scores.append(semantic_score)
                    out_prototypes.append(prototype_id)
                    observed_track_ids.add(track_id)
                out_frame_index.append(int(frame_index[frame_pos]))
                out_image_id.append(int(image_id[frame_pos]))
                out_offsets.append(len(out_track_id))
            output_path = args.output_root / detection_file.name
            np.savez_compressed(
                output_path,
                frame_index=np.asarray(out_frame_index, dtype=np.int64),
                image_id=np.asarray(out_image_id, dtype=np.int64),
                frame_offsets=np.asarray(out_offsets, dtype=np.int64),
                track_id=np.asarray(out_track_id, dtype=np.int64),
                boxes=np.asarray(out_boxes, dtype=np.float32).reshape(-1, 4),
                score=np.asarray(out_scores, dtype=np.float32),
                tracking_score=np.asarray(out_scores, dtype=np.float32),
                semantic_score=np.asarray(out_semantic_scores, dtype=np.float32),
                prototype_id=np.asarray(out_prototypes, dtype=np.int32),
            )
            stats = {
                "video_id": video_id,
                "frames": len(frame_index),
                "detections": int(len(prototypes)),
                "track_rows": len(out_track_id),
                "tracks": len(observed_track_ids),
                "frames_with_detections": frames_with_detections,
                "empty_frames": len(frame_index) - frames_with_detections,
                "high_detections": high,
                "low_detections": low,
                "mean_track_length": (
                    len(out_track_id) / max(len(observed_track_ids), 1)
                ),
            }
        total_frames += stats["frames"]
        total_detections += stats["detections"]
        total_track_rows += stats["track_rows"]
        total_tracks += stats["tracks"]
        total_frames_with_detections += stats["frames_with_detections"]
        total_empty_frames += stats["empty_frames"]
        if video_index < args.sanity_videos:
            sanity.append(stats)
        if video_index % 20 == 0:
            print(stats, flush=True)

    result = {
        "status": "PASS",
        "association": "category-agnostic ByteTrack; only bbox/score/frame used",
        "track_thresh": args.track_thresh,
        "low_thresh": args.low_thresh,
        "match_thresh": args.match_thresh,
        "track_buffer": args.track_buffer,
        "frame_rate": args.frame_rate,
        "score_key": args.score_key,
        "new_track_thresh": args.new_track_thresh,
        "filter_mot_aspect": args.filter_mot_aspect,
        "videos": len(files),
        "frames": total_frames,
        "frames_with_detections": total_frames_with_detections,
        "empty_frames": total_empty_frames,
        "detections": total_detections,
        "track_rows": total_track_rows,
        "tracks": total_tracks,
        "mean_track_length": total_track_rows / max(total_tracks, 1),
        "wall_seconds": time.time() - start,
        "sanity_first_20_no_gt": sanity,
    }
    (args.output_root / "bytetrack_runtime.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
