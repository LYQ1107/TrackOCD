#!/usr/bin/env python3
"""Filter a full native replay to the frozen 91-video event subset.

This is a storage/reproducibility helper only: it preserves every row of the
selected videos, does not use labels, and writes atomically.
"""
from __future__ import annotations

import argparse, json, os, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVENT_REPLAY = ROOT / "outputs/iclr27_phase82r/replays/temporal_app_mean_r1.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--input", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    vids = {int(json.loads(line)["video_id"]) for line in EVENT_REPLAY.open(encoding="utf-8") if line.strip()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=str(args.output.parent))
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for line in args.input.open(encoding="utf-8"):
                if not line.strip() or int(json.loads(line)["video_id"]) not in vids: continue
                out.write(line); count += 1
            out.flush(); os.fsync(out.fileno())
        os.replace(tmp, args.output)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "video_count": len(vids), "row_count": count}, sort_keys=True))


if __name__ == "__main__": main()
