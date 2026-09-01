"""Build a COCO-format TAO JSON for an arbitrary video shard."""
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
    all_data = json.loads(ALL_JSON.read_text())
    videos = [v for v in all_data["videos"] if v["id"] in ids]
    if len(videos) != len(ids):
        missing = ids - {v["id"] for v in videos}
        raise SystemExit(f"missing videos: {sorted(missing)}")
    vid_set = set(ids)
    images = [im for im in all_data["images"] if im["video_id"] in vid_set]
    img_ids = {im["id"] for im in images}
    anns = [a for a in all_data["annotations"] if a["image_id"] in img_ids]
    subset = {
        "info": all_data.get("info", {}),
        "licenses": all_data.get("licenses", []),
        "categories": all_data.get("categories", []),
        "videos": videos, "images": images, "annotations": anns,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(subset, separators=(",", ":")))
    print("wrote", args.out, "videos", len(videos), "images", len(images),
          "anns", len(anns))


if __name__ == "__main__":
    main()
