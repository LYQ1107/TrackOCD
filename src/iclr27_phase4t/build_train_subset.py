"""Build a TAO-train subset JSON (OVTR validation format) for frozen-Q1
inference. Videos are selected to cover all 48 supported-known categories;
no benchmark-novel semantic labels are used downstream (only 48-known
category ids drive pseudo-novel supervision; novel-role GT boxes are
ignored for semantic roles and only usable as class-agnostic validity,
which this phase conservatively excludes).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TRAIN = ROOT / "data" / "raw" / "tao" / "annotations" / "train.json"
VAL_TMPL = ROOT / "third_party" / "research_refs_phase4n" / "OVTR" / "data" / "validation_ours_v1.json"
OUT = ROOT / "third_party" / "research_refs_phase4n" / "OVTR" / "data" / "train_subset_ours_v1.json"
KNOWN = set(json.loads((ROOT / "data" / "trackocd_v1" / "pure" / "splits" / "supported_known_ids.json").read_text()))


def main(n_videos: int = 60, seed: int = 20260814) -> None:
    d = json.loads(TRAIN.read_text())
    vids = {v["id"]: v for v in d["videos"]}
    imgs_by_vid: dict[int, list[dict]] = defaultdict(list)
    for im in d["images"]:
        imgs_by_vid[int(im["video_id"])].append(im)
    anns_by_vid: dict[int, list[dict]] = defaultdict(list)
    for a in d["annotations"]:
        if a.get("iscrowd"):
            continue
        anns_by_vid[int(a["video_id"])].append(a)

    known_cats_by_vid: dict[int, set[int]] = {}
    n_known_anns: Counter[int] = Counter()
    for vid, anns in anns_by_vid.items():
        cats = {int(a["category_id"]) for a in anns}
        known_cats_by_vid[vid] = cats & KNOWN
        n_known_anns[vid] = sum(1 for a in anns if int(a["category_id"]) in KNOWN)

    # greedy cover of all 48 known categories, then top-up by known-annotation count
    remaining = set(KNOWN)
    chosen: list[int] = []
    pool = sorted(vids, key=lambda v: -n_known_anns[v])
    for vid in pool:
        if remaining & known_cats_by_vid[vid]:
            chosen.append(vid)
            remaining -= known_cats_by_vid[vid]
        if not remaining:
            break
    for vid in pool:
        if len(chosen) >= n_videos:
            break
        if vid not in chosen:
            chosen.append(vid)
    chosen = set(chosen)
    print("selected videos:", len(chosen), "known categories covered:",
          len(KNOWN - {c for v in chosen for c in known_cats_by_vid[v]}))

    tmpl = json.loads(VAL_TMPL.read_text())
    videos = []
    images = []
    anns = []
    for vid in sorted(chosen):
        v = vids[vid]
        videos.append({"id": v["id"], "name": v["name"],
                       "width": v["width"], "height": v["height"]})
        ims = sorted(imgs_by_vid[vid], key=lambda x: int(x["frame_index"]))
        for rank, im in enumerate(ims):
            images.append({
                "id": im["id"], "video": im["video"],
                "width": im["width"], "height": im["height"],
                "file_name": im["file_name"],
                "frame_index": im["frame_index"],
                "frame_id": rank,
                "video_id": vid,
                "license": im.get("license", 0),
            })
        for a in anns_by_vid[vid]:
            anns.append({
                "segmentation": a.get("segmentation"),
                "bbox": a["bbox"], "area": a.get("area"),
                "iscrowd": 0, "id": a["id"], "image_id": a["image_id"],
                "category_id": a["category_id"], "track_id": a.get("track_id"),
                "video_id": vid, "instance_id": a.get("track_id"),
            })
    out = {
        "info": tmpl.get("info", {}),
        "licenses": tmpl.get("licenses", []),
        "categories": tmpl["categories"],
        "videos": videos,
        "images": images,
        "annotations": anns,
        "tracks": tmpl.get("tracks", []),
    }
    OUT.write_text(json.dumps(out))
    print("wrote", OUT, "images", len(images), "anns", len(anns))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=60)
    args = ap.parse_args()
    main(args.n_videos)
