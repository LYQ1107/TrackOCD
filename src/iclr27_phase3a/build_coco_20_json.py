#!/usr/bin/env python3
"""Build a COCO-format TAO subset JSON for the deterministic 20 videos.

The original SimOWT online inference path uses a COCO-style dataset
(`coco_2017_val_agn` -> tao/frames + tao/annotations/val_split/all.json)
and the `DetrDatasetMapper`.  Reusing that path is the correct way to run
the instrumented writer, instead of forcing the YTVIS-style mapper.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT_DIR = PROJECT_ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset"
ALL_JSON = PROJECT_ROOT / "third_party" / "SimOWT" / "datasets" / "tao" / "annotations" / "val_split" / "all.json"
SELECTED_CSV = PROJECT_ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "selected_20_videos.csv"
TARGET = OUT_DIR / "validation_20_coco.json"


def main() -> None:
    selected_ids = set()
    with open(SELECTED_CSV, newline="") as f:
        for row in csv.DictReader(f):
            selected_ids.add(int(row["video_id"]))

    with open(ALL_JSON) as f:
        all_data = json.load(f)

    videos = [v for v in all_data["videos"] if v["id"] in selected_ids]
    video_ids = {v["id"] for v in videos}
    if len(videos) != 20 or video_ids != selected_ids:
        raise SystemExit(
            f"selected video mismatch: got {len(videos)} videos, "
            f"expected ids {sorted(selected_ids)}"
        )

    images = [im for im in all_data["images"] if im["video_id"] in video_ids]
    image_ids = {im["id"] for im in images}
    annotations = [a for a in all_data["annotations"] if a["image_id"] in image_ids]

    # Frame order inside a video must be preserved.  The COCO loader sorts by
    # image id, so keep the original ids (they are monotonic per video).
    by_video: dict[int, list[dict]] = {}
    for im in images:
        by_video.setdefault(im["video_id"], []).append(im)
    for vid, ims in by_video.items():
        frames = [im["frame_index"] for im in ims]
        if frames != sorted(frames):
            raise SystemExit(f"video {vid}: frame_index not monotonic in source order")

    subset = {
        "info": all_data.get("info", {}),
        "licenses": all_data.get("licenses", []),
        "categories": all_data.get("categories", []),
        "videos": videos,
        "images": images,
        "annotations": annotations,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="validation_20_coco_", suffix=".json", dir=OUT_DIR)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(subset, f, separators=(",", ":"))
        os.replace(tmp, TARGET)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    print(f"wrote {TARGET}")
    print(f"videos={len(videos)} images={len(images)} annotations={len(annotations)}")
    for v in videos:
        print(v["id"], v["name"])


if __name__ == "__main__":
    main()
