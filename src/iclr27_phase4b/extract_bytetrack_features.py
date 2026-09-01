"""Extract DINOv2 track-mean features for ByteTrack tracks (shardable)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(ROOT))

from src.features.extract import (
    load_dinov2, make_crop_transform, extract_track_features,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.stream) if l.strip()]
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    print("shard", args.shard, "tracks", len(rows), flush=True)
    model, size = load_dinov2()
    transform = make_crop_transform(size)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    failed = 0
    for row in rows:
        try:
            extract_track_features(
                row, model, size, transform, args.cache_dir,
                clip_preprocess=None, mode="mean", sampling="score",
            )
            done += 1
        except Exception as e:
            failed += 1
            print("ERROR", row["sample_id"], e, flush=True)
    print("shard", args.shard, "done", done, "failed", failed, flush=True)


if __name__ == "__main__":
    main()
