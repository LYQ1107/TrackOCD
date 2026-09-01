"""TAO annotation coverage audit for the frozen 20-video dev subset."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase4s.protocol import (
    DEV_GT_JSON,
    DEV_VIDEOS_CSV,
    TAO_VAL_ANN,
    known_ids,
    load_dev_videos,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/iclr27_phase5b/audit/tao_coverage")
    args = ap.parse_args()

    val = json.loads(TAO_VAL_ANN.read_text())
    dev_videos = set(load_dev_videos())
    known = known_ids()
    vid_of_img = {im["id"]: int(im["video_id"]) for im in val["images"]}
    fr_of_img = {im["id"]: int(im["frame_index"]) for im in val["images"]}

    n_all_imgs = len(val["images"])
    n_dev_imgs = sum(1 for im in val["images"] if int(im["video_id"]) in dev_videos)
    dev_img_ids = {im["id"] for im in val["images"]
                   if int(im["video_id"]) in dev_videos}
    ann = [a for a in val["annotations"] if a.get("image_id") in dev_img_ids]
    ann_no_crowd = [a for a in ann if not a.get("iscrowd")]
    n_ann_frames = len({a["image_id"] for a in ann})
    tracks = defaultdict(list)
    for a in ann_no_crowd:
        tracks[(int(a["video_id"]), int(a["track_id"]))].append(a)
    lens = np.array([len(v) for v in tracks.values()])
    cats = Counter(int(a["category_id"]) for a in ann_no_crowd)
    cats_known = sum(v for c, v in cats.items() if c in known)
    cats_novel = sum(v for c, v in cats.items() if c not in known)

    # frames per video and annotated frames per video
    frames_per_video = defaultdict(int)
    ann_frames_per_video = defaultdict(int)
    for im in val["images"]:
        if int(im["video_id"]) in dev_videos:
            frames_per_video[int(im["video_id"])] += 1
    for a in ann:
        ann_frames_per_video[int(a["video_id"])] += 1

    summary = {
        "n_images_all_val": n_all_imgs,
        "n_videos_val": len({im["video_id"] for im in val["images"]}),
        "dev_video_ids": sorted(dev_videos),
        "n_dev_images_total": n_dev_imgs,
        "n_dev_annotated_frames": n_ann_frames,
        "annotation_frame_ratio": n_ann_frames / max(n_dev_imgs, 1),
        "n_dev_annotations": len(ann),
        "n_dev_non_crowd_annotations": len(ann_no_crowd),
        "n_dev_tracks": len(tracks),
        "track_len_mean": float(lens.mean()),
        "track_len_median": float(np.median(lens)),
        "track_len_max": int(lens.max()),
        "n_categories_dev": len(cats),
        "n_known_category_anns": cats_known,
        "n_novel_category_anns": cats_novel,
        "frames_per_video": dict(frames_per_video),
        "annotated_frames_per_video": dict(ann_frames_per_video),
        "track_length_buckets": {
            "1": int((lens == 1).sum()),
            "2": int((lens == 2).sum()),
            "3": int((lens == 3).sum()),
            "4_10": int(((lens >= 4) & (lens <= 10)).sum()),
            "gt10": int((lens > 10).sum()),
        },
        "top_categories": cats.most_common(20),
        "has_iscrowd": any(a.get("iscrowd") for a in ann),
        "n_iscrowd": sum(1 for a in ann if a.get("iscrowd")),
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("top_categories", "frames_per_video",
                                   "annotated_frames_per_video",
                                   "track_length_buckets")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
