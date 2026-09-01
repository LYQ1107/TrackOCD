"""Convert official TAO train (per-image annotations) to YTVIS-style
video-level JSON consumable by the IDOL training loader.

Output is class-agnostic (single foreground class) but preserves physical
track IDs, per-frame bboxes and segmentations.  This is the legal
class-agnostic physical supervision input for Phase 4P joint training.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
SRC = ROOT / "data" / "raw" / "tao" / "annotations" / "train.json"
OUT = ROOT / "third_party" / "SimOWT" / "datasets" / "tao" / "annotations" \
    / "train_agn_ytvis.json"


def main():
    d = json.load(open(SRC))
    imgs = sorted(d["images"], key=lambda x: (x["video_id"], x["frame_index"]))
    vid_order = []
    for im in imgs:
        if im["video_id"] not in vid_order:
            vid_order.append(im["video_id"])
    vids_by_id = {v["id"]: v for v in d["videos"]}

    # Per (video, track): list of (frame_index, bbox, segmentation)
    imgid_to_meta = {im["id"]: (im["video_id"], im["frame_index"],
                                im["file_name"]) for im in imgs}
    by_track = defaultdict(list)
    for a in d["annotations"]:
        if a.get("iscrowd", 0):
            continue
        _vid, frame_index, fname = imgid_to_meta[a["image_id"]]
        by_track[(a["video_id"], a["track_id"])].append((
            fname, frame_index, a["bbox"], a.get("segmentation")))

    vid2frame = defaultdict(dict)  # video_id -> {frame_index: file_name}
    vid2file = defaultdict(list)
    for im in imgs:
        vid2frame[im["video_id"]][im["frame_index"]] = im["file_name"]
    for v in d["videos"]:
        vid2file[v["id"]] = [f for _, f in sorted(
            vid2frame[v["id"]].items())]

    videos = []
    images = []
    anns = []
    ann_id = 0
    for vid in vid_order:
        v = vids_by_id[vid]
        files = vid2file[vid]
        videos.append({
            "id": vid, "name": v["name"], "width": v["width"],
            "height": v["height"], "length": len(files),
            "file_names": files,
        })
        for im in imgs:
            if im["video_id"] != vid:
                continue
            images.append({
                "id": im["id"], "video_id": vid,
                "file_name": im["file_name"],
                "frame_index": im["frame_index"],
                "width": im["width"], "height": im["height"],
            })
        # anns per track: bboxes/segmentations aligned to file order
        file_to_frame = {f: fi for fi, f in enumerate(files)}
        for (vid_id, track_id), items in sorted(by_track.items()):
            if vid_id != vid:
                continue
            n = len(files)
            bboxes = [None] * n
            segms = [None] * n
            for fname, _fi, bbox, segm in items:
                fidx = file_to_frame.get(fname)
                if fidx is None:
                    continue
                bboxes[fidx] = bbox
                segms[fidx] = segm
            anns.append({
                "id": ann_id, "video_id": vid, "category_id": 1,
                "track_id": track_id, "iscrowd": 0,
                "bboxes": bboxes, "segmentations": segms,
            })
            ann_id += 1
    out = {
        "info": d.get("info", {}),
        "licenses": d.get("licenses", []),
        "categories": [{"id": 1, "name": "object", "isthing": 1,
                        "color": [255, 255, 255]}],
        "videos": videos, "images": images, "annotations": anns,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print("WROTE", OUT)
    print("videos", len(videos), "images", len(images), "anns", len(anns))


if __name__ == "__main__":
    main()
