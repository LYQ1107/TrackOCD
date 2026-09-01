#!/usr/bin/env python3
"""Convert per-image SimOWT trajectory JSONs into a flat TAO tracker JSON."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-json", required=True, type=Path)
    args = ap.parse_args()

    annotations = []
    for p in sorted(args.input_dir.glob("*.json")):
        annotations.extend(json.loads(p.read_text()))
    annotations.sort(key=lambda a: (a["video_id"], a["image_id"], a["track_id"]))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tracker_", suffix=".json", dir=args.output_json.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(annotations, f, separators=(",", ":"))
        os.replace(tmp, args.output_json)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"wrote {len(annotations)} annotations -> {args.output_json}")


if __name__ == "__main__":
    main()
