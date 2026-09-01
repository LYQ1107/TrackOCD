"""Split the resume video set into balanced COCO-format shards for parallel
SimOWT export across GPUs."""
from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
ALL_JSON = ROOT / "third_party/SimOWT/datasets/tao/annotations/val_split/all.json"
OUT_DIR = ROOT / "outputs" / "iclr27_phase3b" / "full_export" / "tao_subset"


def main():
    num_shards = int(os.environ.get("PHASE3B_SHARDS", "8"))
    state = json.load(open(ROOT / "runs/iclr27_phase3b/resume_state.json"))
    resume_videos = set(state["resume_videos"])
    all_data = json.load(open(ALL_JSON))
    frames_by_video = defaultdict(int)
    for im in all_data["images"]:
        if im["video_id"] in resume_videos:
            frames_by_video[im["video_id"]] += 1
    # greedy balanced assignment by frame count
    shard_frames = [0] * num_shards
    shard_videos = [[] for _ in range(num_shards)]
    for vid in sorted(frames_by_video, key=lambda v: -frames_by_video[v]):
        idx = min(range(num_shards), key=lambda i: shard_frames[i])
        shard_videos[idx].append(vid)
        shard_frames[idx] += frames_by_video[vid]
    image_by_vid = defaultdict(list)
    for im in all_data["images"]:
        if im["video_id"] in resume_videos:
            image_by_vid[im["video_id"]].append(im)
    resume_image_ids = {im["id"] for ims in image_by_vid.values() for im in ims}
    ann_by_img = defaultdict(list)
    for a in all_data["annotations"]:
        if a["image_id"] in resume_image_ids:
            ann_by_img[a["image_id"]].append(a)
    for i, vids in enumerate(shard_videos):
        vids = sorted(vids)
        images = [im for vid in vids for im in sorted(image_by_vid[vid], key=lambda x: x["id"])]
        image_ids = {im["id"] for im in images}
        anns = [a for img_id in sorted(image_ids) for a in ann_by_img[img_id]]
        subset = {
            "info": all_data.get("info", {}),
            "licenses": all_data.get("licenses", []),
            "categories": all_data.get("categories", []),
            "videos": [v for v in all_data["videos"] if v["id"] in vids],
            "images": images,
            "annotations": anns,
        }
        out = OUT_DIR / f"validation_resume_shard_{i:02d}.json"
        fd, tmp = tempfile.mkstemp(prefix="shard_", suffix=".json", dir=OUT_DIR)
        with os.fdopen(fd, "w") as f:
            json.dump(subset, f, separators=(",", ":"))
        os.replace(tmp, out)
        print(i, "videos", len(vids), "frames", shard_frames[i], "->", out)
    print("total resume videos", len(resume_videos), "frames", sum(shard_frames))


if __name__ == "__main__":
    main()
