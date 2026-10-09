#!/usr/bin/env python3
"""Resumable NMS + fixed ByteTrack; no GT is loaded by this command."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.baselines.pandas_bytetrack.diagnostics import atomic_json, read_video, sha256, summary
from src.baselines.pandas_bytetrack.tracking_postprocess import track_video
from src.iclr27_phase3b.bytetrack import BYTETracker


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detections-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--budget-root", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--subset", choices=["calibration", "confirmation", "full"], required=True)
    p.add_argument("--variant", choices=["Dedup-A", "Dedup-B", "Dedup-C"], required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    selection = json.loads(args.selection.read_text())
    config = selection["variants"][args.variant]
    videos = selection[args.subset+"_ids"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    stats, all_counts, length_hist = [], [], {}
    start = time.time()
    for vid in videos:
        used = sum(x.stat().st_size for x in args.budget_root.rglob("*") if x.is_file() and not x.is_symlink())
        if used >= 10 * 1024**3:
            raise RuntimeError("10 GiB additional-output soft budget reached; audit before more writes")
        src = args.detections_root / f"video_{vid:04d}.npz"
        dst = args.output_root / src.name
        marker = dst.with_suffix(".json")
        src_hash = sha256(src)
        if dst.exists() and marker.exists():
            st = json.loads(marker.read_text())
            if st["source_sha256"] != src_hash or st["config"] != config or st["output_sha256"] != sha256(dst):
                raise ValueError(f"resume validation failed: {dst}")
        else:
            if dst.exists() or marker.exists():
                raise RuntimeError(f"incomplete output pair: {dst}; inspect explicitly before recovery")
            data = read_video(src)
            tracker = BYTETracker(track_thresh=.5, low_thresh=.1, new_track_thresh=.6,
                                  match_thresh=.8, track_buffer=30, frame_rate=30,
                                  filter_mot_aspect=False)
            tracker.reset()
            result, counts, lengths = track_video(data, tracker, **config)
            tmp = dst.with_suffix(".npz.tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(f, **result)
            os.replace(tmp, dst)
            if sha256(src) != src_hash:
                raise ValueError("frozen input changed during tracking")
            values, hist = np.unique(lengths, return_counts=True)
            st = {"status": "PASS", "video_id": vid, "config": config,
                  "frames": len(counts), "input_detections": len(data["boxes"]),
                  "candidates": int(counts.sum()), "counts_histogram": np.bincount(counts).tolist(),
                  "track_rows": len(result["track_id"]), "physical_tracks": len(lengths),
                  "track_length_histogram": {str(int(k)): int(v) for k,v in zip(values,hist)},
                  "source_sha256": src_hash, "output_sha256": sha256(dst)}
            atomic_json(marker, st)
        stats.append(st)
        for count, freq in enumerate(st["counts_histogram"]):
            all_counts.extend([count]*freq)
        for k,v in st["track_length_histogram"].items():
            length_hist[k] = length_hist.get(k, 0) + v
        print(json.dumps({k:st[k] for k in ["video_id", "frames", "candidates", "physical_tracks"]}), flush=True)
    lengths = np.concatenate([np.full(v, int(k)) for k,v in length_hist.items()]) if length_hist else np.empty(0)
    result = {"status": "PASS", "variant": args.variant, "subset": args.subset,
              "config": config, "video_ids": videos, "videos": len(stats),
              "frames": sum(s["frames"] for s in stats),
              "input_detections": sum(s["input_detections"] for s in stats),
              "candidate_counts": summary(all_counts),
              "track_rows": sum(s["track_rows"] for s in stats),
              "physical_tracks": sum(s["physical_tracks"] for s in stats),
              "track_lengths": summary(lengths), "elapsed_seconds": time.time()-start,
              "source_metadata_policy": "nearest IoU to selected candidate, approximate; metadata never enters association",
              "bytetrack": {"track_thresh": .5, "low_thresh": .1, "new_track_thresh": .6,
                            "match_thresh": .8, "track_buffer": 30, "frame_rate": 30,
                            "filter_mot_aspect": False, "score_key": "foreground_scores"}}
    atomic_json(args.output_root / "runtime.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
