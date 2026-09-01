"""Build a COCO-format resume dataset starting at the first unfinished video.

The SimOWT tracker state is per-video, so resuming mid-video is invalid.
We find the video containing the next unprocessed image and restart from
that video's first frame; all fully completed earlier videos are kept.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
ALL_JSON = ROOT / "third_party/SimOWT/datasets/tao/annotations/val_split/all.json"
OUT = ROOT / "outputs" / "iclr27_phase3b" / "full_export" / "tao_subset" / "validation_resume_coco.json"


def main():
    all_data = json.load(open(ALL_JSON))
    images = sorted(all_data["images"], key=lambda x: x["id"])
    processed = 13710  # from last completed log line before the interruption
    if processed >= len(images):
        raise SystemExit("no resume needed")
    next_idx = processed
    partial_video = images[next_idx]["video_id"]
    # group video -> first/last image positions (processing order is image id order)
    vid_pos = defaultdict(list)
    for i, im in enumerate(images):
        vid_pos[im["video_id"]].append(i)
    resume_videos = set()
    for vid, positions in vid_pos.items():
        if min(positions) >= vid_pos[partial_video][0] or vid == partial_video:
            resume_videos.add(vid)
    resume_videos = sorted(resume_videos)
    resume_images = [im for im in images if im["video_id"] in resume_videos]
    image_ids = {im["id"] for im in resume_images}
    resume_anns = [a for a in all_data["annotations"] if a["image_id"] in image_ids]
    resume_vid_meta = [v for v in all_data["videos"] if v["id"] in resume_videos]
    subset = {
        "info": all_data.get("info", {}),
        "licenses": all_data.get("licenses", []),
        "categories": all_data.get("categories", []),
        "videos": resume_vid_meta,
        "images": resume_images,
        "annotations": resume_anns,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="resume_", suffix=".json", dir=OUT.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(subset, f, separators=(",", ":"))
    os.replace(tmp, OUT)
    print("partial_video", partial_video)
    print("resume_videos", len(resume_videos), "first", resume_videos[:5], "last", resume_videos[-5:])
    print("resume_images", len(resume_images), "->", OUT)
    # write a small state file for the launcher
    (ROOT / "runs/iclr27_phase3b/resume_state.json").write_text(json.dumps({
        "processed_log_frames": processed,
        "partial_video": partial_video,
        "resume_videos": resume_videos,
        "resume_images": len(resume_images),
    }, indent=1))


if __name__ == "__main__":
    main()
