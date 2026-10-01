"""Run ByteTrack on a frozen pre-association detection stream."""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase3b.bytetrack import BYTETracker


FRAME_RE = re.compile(r"(\d+)$")


def load_detection_frames(det_file: Path):
    frames = defaultdict(list)
    for line in det_file.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        frames[r["frame_order"]].append(r)
    return frames


def load_tao_frame_manifest(annotation_path: Path, frame_root: Path, video_id: int):
    """Enumerate every real TAO frame, including frames with no detections."""
    data = json.loads(annotation_path.read_text())
    video = next(video for video in data["videos"] if int(video["id"]) == video_id)
    image_id_by_path = {
        str((frame_root / image["file_name"]).resolve()): int(image["id"])
        for image in data["images"]
        if int(image["video_id"]) == video_id
    }
    video_dir = frame_root / video["name"]
    paths = sorted(
        video_dir.glob("*.jpg"),
        key=lambda path: (
            path.stem[: -len(FRAME_RE.search(path.stem).group(1))]
            if FRAME_RE.search(path.stem) else path.stem,
            int(FRAME_RE.search(path.stem).group(1))
            if FRAME_RE.search(path.stem) else -1,
        ),
    )
    if not paths:
        raise RuntimeError(f"no TAO frames found for video {video_id}: {video_dir}")
    return [
        {
            "frame_order": frame_order,
            "image_id": image_id_by_path.get(str(path.resolve()), -1),
        }
        for frame_order, path in enumerate(paths)
    ]


def run_video(video_id, det_file, frame_manifest, out_dir, tracker, score_key):
    detections_by_frame = load_detection_frames(det_file)
    tracker.reset()
    n_high = 0
    n_low = 0
    n_low_triggers = 0
    frames_with_detections = 0
    for frame_meta in frame_manifest:
        recs = detections_by_frame.get(frame_meta["frame_order"], [])
        dets = []
        for r in recs:
            if score_key not in r:
                raise KeyError(f"missing tracking score {score_key} in {det_file}")
            tracking_score = float(r[score_key])
            if not np.isfinite(tracking_score) or not 0.0 <= tracking_score <= 1.0:
                raise ValueError(f"tracking score out of range: {tracking_score}")
            x1, y1, x2, y2 = r["bbox_xyxy_original"]
            dets.append([x1, y1, x2, y2, tracking_score])
            if tracking_score > tracker.track_thresh:
                n_high += 1
            elif tracking_score > tracker.low_thresh:
                n_low += 1
        if recs:
            frames_with_detections += 1
        if any(tracker.track_thresh >= float(r[score_key]) > tracker.low_thresh for r in recs):
            n_low_triggers += 1
        out = tracker.update(dets)
        frame_preds = []
        for row in out:
            x1, y1, x2, y2, tid, score = row.tolist()
            pred = {
                "bbox": [int(round(x1)), int(round(y1)),
                         int(round(x2 - x1)), int(round(y2 - y1))],
                "track_id": int(tid),
                "category_id": 1,
                "image_id": frame_meta["image_id"],
                "video_id": video_id,
                "score": float(score),
            }
            frame_preds.append(pred)
        if frame_preds and frame_meta["image_id"] >= 0:
            pid = str(frame_meta["image_id"]).zfill(10)
            target = out_dir / f"{pid}.json"
            fd, tmp = tempfile.mkstemp(prefix="bt_", suffix=".json", dir=out_dir)
            with os.fdopen(fd, "w") as f:
                json.dump(frame_preds, f, separators=(",", ":"))
            os.replace(tmp, target)
    return {
        "frames": len(frame_manifest),
        "frames_with_detections": frames_with_detections,
        "empty_frames": len(frame_manifest) - frames_with_detections,
        "n_high": n_high,
        "n_low": n_low,
        "low_stage_frames": n_low_triggers,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--tao-annotation", required=True, type=Path)
    ap.add_argument("--tao-frame-root", required=True, type=Path)
    ap.add_argument("--score-key", default="foreground_score")
    ap.add_argument("--track-thresh", type=float, default=0.5)
    ap.add_argument("--low-thresh", type=float, default=0.1)
    ap.add_argument("--match-thresh", type=float, default=0.8)
    ap.add_argument("--track-buffer", type=int, default=30)
    ap.add_argument("--frame-rate", type=int, default=30)
    ap.add_argument("--new-track-thresh", type=float, default=None)
    ap.add_argument("--filter-mot-aspect", action="store_true")
    ap.add_argument("--runtime-json", type=Path, default=None)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracker = BYTETracker(track_thresh=args.track_thresh, low_thresh=args.low_thresh,
                          match_thresh=args.match_thresh, track_buffer=args.track_buffer,
                          frame_rate=args.frame_rate,
                          new_track_thresh=args.new_track_thresh,
                          filter_mot_aspect=args.filter_mot_aspect)
    stats = {}
    t0 = time.time()
    for det_file in sorted(args.detections_dir.glob("*.jsonl")):
        vid = int(det_file.stem)
        frame_manifest = load_tao_frame_manifest(args.tao_annotation, args.tao_frame_root, vid)
        s = run_video(vid, det_file, frame_manifest, args.output_dir, tracker, args.score_key)
        stats[vid] = s
        print(vid, s, flush=True)
    runtime = {
        "wall_seconds": time.time() - t0,
        "videos": len(stats),
        "score_key": args.score_key,
        "new_track_thresh": args.new_track_thresh,
        "filter_mot_aspect": args.filter_mot_aspect,
    }
    runtime_path = args.runtime_json or (args.output_dir.parent / "runtime.json")
    runtime_path.write_text(json.dumps(runtime, indent=1))
    print(json.dumps(runtime))


if __name__ == "__main__":
    main()
