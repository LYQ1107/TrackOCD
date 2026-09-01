"""Convert per-image Phase 4I trajectory JSONs into TrackEval TAO format."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--tracker-name", required=True)
    ap.add_argument("--output-root", required=True, type=Path)
    args = ap.parse_args()
    out = args.output_root / args.tracker_name / "data" / "pred.json"
    annotations = []
    for p in sorted(args.input_dir.glob("*.json")):
        if p.name == "trackeval.json":
            continue
        annotations.extend(json.loads(p.read_text()))
    annotations.sort(key=lambda a: (a["video_id"], a["image_id"],
                                    a["track_id"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tracker_", suffix=".json",
                               dir=out.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(annotations, f, separators=(",", ":"))
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"wrote {len(annotations)} annotations -> {out}")


if __name__ == "__main__":
    main()
