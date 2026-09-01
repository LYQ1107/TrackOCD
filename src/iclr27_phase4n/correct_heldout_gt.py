"""Rebuild the Phase 4L held-out GT with real TAO categories.

Phase 4L built the held-out GT from
third_party/SimOWT/datasets/tao/annotations/val_split/all.json, in which
all categories were collapsed to id 1.  This script rebuilds the same
24-video subset from the original TAO validation.json (1230 categories)
with the same largest-box-per-(image,track) cleanup, so that known/novel
routing and memory audits are meaningful on held-out.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
ALL_JSON = ROOT / "third_party" / "SimOWT" / "datasets" / "tao" / \
    "annotations" / "validation.json"
HELDOUT_CSV = ROOT / "outputs" / "iclr27_phase4l" / "heldout" / \
    "selected_heldout_videos.csv"
OUT = ROOT / "outputs" / "iclr27_phase4n" / "audit" / \
    "validation_heldout_tao_corrected.json"


def main():
    d = json.loads(ALL_JSON.read_text())
    ids = {int(r["video_id"]) for r in csv.DictReader(open(HELDOUT_CSV))}
    videos = [v for v in d["videos"] if v["id"] in ids]
    images = [im for im in d["images"] if im["video_id"] in ids]
    img_ids = {im["id"] for im in images}
    anns = [a for a in d["annotations"] if a["image_id"] in img_ids]
    best = {}
    for a in anns:
        key = (a["image_id"], a["track_id"])
        area = float(a.get("area", 0.0))
        if key not in best or area > best[key][0]:
            best[key] = (area, a)
    anns = [a for _, a in best.values()]
    track_ids = {a["track_id"] for a in anns}
    tracks = [t for t in d["tracks"] if t["id"] in track_ids]
    subset = {k: d.get(k) for k in ("info", "licenses", "categories")}
    subset.update({"videos": videos, "images": images, "tracks": tracks,
                   "annotations": anns})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(subset, separators=(",", ":")))
    from collections import Counter
    c = Counter(a["category_id"] for a in anns)
    print("CORRECTED_HELDOUT_GT", len(videos), len(images), len(anns),
          "categories", len(c))


if __name__ == "__main__":
    main()
