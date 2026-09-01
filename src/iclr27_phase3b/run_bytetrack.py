"""Run ByteTrack on a frozen pre-association detection stream."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase3b.bytetrack import BYTETracker


def load_frames(det_file: Path):
    frames = defaultdict(list)
    for line in det_file.read_text().splitlines():
        r = json.loads(line)
        frames[r["frame_order"]].append(r)
    return [frames[k] for k in sorted(frames)]


def run_video(video_id, det_file, out_dir, tracker):
    frames = load_frames(det_file)
    tracker.reset()
    n_high = 0
    n_low = 0
    n_low_triggers = 0
    for fi, recs in enumerate(frames):
        dets = []
        for r in recs:
            x1, y1, x2, y2 = r["bbox_xyxy_original"]
            dets.append([x1, y1, x2, y2, r["score"]])
            if r["score"] > tracker.track_thresh:
                n_high += 1
            elif r["score"] > tracker.low_thresh:
                n_low += 1
        if any(tracker.track_thresh >= r["score"] > tracker.low_thresh for r in recs):
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
                "image_id": recs[0]["image_id"] if recs else None,
                "video_id": video_id,
                "score": float(score),
            }
            frame_preds.append(pred)
        if frame_preds:
            pid = str(frame_preds[0]["image_id"]).zfill(10)
            target = out_dir / f"{pid}.json"
            fd, tmp = tempfile.mkstemp(prefix="bt_", suffix=".json", dir=out_dir)
            with os.fdopen(fd, "w") as f:
                json.dump(frame_preds, f, separators=(",", ":"))
            os.replace(tmp, target)
    return {"frames": len(frames), "n_high": n_high, "n_low": n_low,
            "low_stage_frames": n_low_triggers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--track-thresh", type=float, default=0.5)
    ap.add_argument("--low-thresh", type=float, default=0.1)
    ap.add_argument("--match-thresh", type=float, default=0.8)
    ap.add_argument("--track-buffer", type=int, default=30)
    ap.add_argument("--frame-rate", type=int, default=30)
    ap.add_argument("--runtime-json", type=Path, default=None)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracker = BYTETracker(track_thresh=args.track_thresh, low_thresh=args.low_thresh,
                          match_thresh=args.match_thresh, track_buffer=args.track_buffer,
                          frame_rate=args.frame_rate)
    stats = {}
    t0 = time.time()
    for det_file in sorted(args.detections_dir.glob("*.jsonl")):
        vid = int(det_file.stem)
        s = run_video(vid, det_file, args.output_dir, tracker)
        stats[vid] = s
        print(vid, s, flush=True)
    runtime = {"wall_seconds": time.time() - t0, "videos": len(stats)}
    runtime_path = args.runtime_json or (args.output_dir.parent / "runtime.json")
    runtime_path.write_text(json.dumps(runtime, indent=1))
    print(json.dumps(runtime))


if __name__ == "__main__":
    main()
