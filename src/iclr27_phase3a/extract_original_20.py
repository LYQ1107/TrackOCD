#!/usr/bin/env python3
"""Extract the legacy SimOWT predictions (O) for the deterministic 20 videos.

Read-only extraction from outputs/simowt/val_predictions.json.  The output
mirrors the per-image writer layout used by the instrumented online run so
that O vs I can be compared frame by frame.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
LEGACY = PROJECT_ROOT / "outputs" / "simowt" / "val_predictions.json"
SELECTED_CSV = PROJECT_ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "selected_20_videos.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "iclr27_phase3a" / "trajectories" / "original_20"


def main() -> None:
    with open(SELECTED_CSV, newline="") as f:
        selected = {int(row["video_id"]) for row in csv.DictReader(f)}
    if len(selected) != 20:
        raise SystemExit(f"expected 20 selected video ids, got {len(selected)}")

    print("loading legacy predictions (read-only)...")
    with open(LEGACY) as f:
        records = json.load(f)

    per_image: dict[int, list[dict]] = defaultdict(list)
    n_kept = 0
    for rec in records:
        vid = rec.get("video_id")
        if vid in selected:
            per_image[rec["image_id"]].append(rec)
            n_kept += 1

    print(f"legacy records total={len(records)} kept={n_kept} frames={len(per_image)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_id, preds in sorted(per_image.items()):
        target = OUT_DIR / f"{str(image_id).zfill(10)}.json"
        fd, tmp = tempfile.mkstemp(prefix="orig_", suffix=".json", dir=OUT_DIR)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(preds, f, separators=(",", ":"))
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    print(f"wrote {len(per_image)} per-image files to {OUT_DIR}")


if __name__ == "__main__":
    main()
