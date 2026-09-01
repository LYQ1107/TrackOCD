"""TAO-format GT JSON for the Phase 4L held-out subset (offline only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
ALL_JSON = ROOT / "third_party" / "SimOWT" / "datasets" / "tao" / \
    "annotations" / "val_split" / "all.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-ids", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    ids = {int(v) for v in args.video_ids.split(",") if v.strip()}
    d = json.loads(ALL_JSON.read_text())
    videos = [v for v in d["videos"] if v["id"] in ids]
    images = [im for im in d["images"] if im["video_id"] in ids]
    img_ids = {im["id"] for im in images}
    anns = [a for a in d["annotations"] if a["image_id"] in img_ids]
    # TAO raw annotations can contain more than one box per (image,
    # track); TrackEval requires unique track ids per timestep.  Keep the
    # largest-area box per (image_id, track_id) as a transparent cleanup.
    best_ann = {}
    for a in anns:
        key = (a["image_id"], a["track_id"])
        area = float(a.get("area", 0.0))
        if key not in best_ann or area > best_ann[key][0]:
            best_ann[key] = (area, a)
    anns = [a for _, a in best_ann.values()]
    track_ids = {a["track_id"] for a in anns}
    tracks = [t for t in d["tracks"] if t["id"] in track_ids]
    subset = {k: d.get(k) for k in ("info", "licenses", "categories")}
    subset.update({"videos": videos, "images": images, "tracks": tracks,
                   "annotations": anns})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(subset, separators=(",", ":")))
    print("wrote", args.out, "videos", len(videos), "images", len(images),
          "tracks", len(tracks), "anns", len(anns))


if __name__ == "__main__":
    main()
