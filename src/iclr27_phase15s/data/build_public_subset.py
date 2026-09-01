"""Materialise a bounded public-TRAIN annotation view for frozen DSCT.

Only public annotation metadata is selected; the physical detector consumes
images/frames and never receives category labels.  The output is a new JSON
under the Phase15S data directory and is written atomically.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def build(annotation: Path, roles_path: Path, out: Path, max_frames: int = 12000) -> dict:
    src = json.loads(annotation.read_text())
    roles = json.loads(roles_path.read_text())["roles"]
    wanted = sorted(set(roles["known_bank_train"]) | set(roles["known_calibration"]) |
                    set(roles["known_audit"]))
    images = [i for i in src.get("images", []) if int(i.get("video_id", -1)) in set(wanted)]
    images.sort(key=lambda i: (int(i["video_id"]), int(i.get("frame_index", i.get("frame_id", 0))), int(i["id"])))
    # The budget is on distinct frame images, and selection is deterministic:
    # retain the earliest bounded frames per video, preserving chronology.
    if len(images) > max_frames:
        keep = set()
        per_video = {}
        for i in images:
            per_video.setdefault(int(i["video_id"]), []).append(i)
        # Proportional allocation with at least one image per selected video.
        remaining = max_frames
        selected = []
        for v in sorted(per_video):
            n = min(len(per_video[v]), max(1, max_frames // max(len(per_video), 1)))
            selected.extend(per_video[v][:n]); remaining -= n
        leftovers = [i for i in images if i not in selected]
        selected.extend(leftovers[:max(0, remaining)])
        images = sorted(selected[:max_frames], key=lambda i: (int(i["video_id"]), int(i.get("frame_index", i.get("frame_id", 0))), int(i["id"])))
    image_ids = {int(i["id"]) for i in images}
    videos = [v for v in src.get("videos", []) if int(v.get("id", -1)) in set(wanted)]
    videos.sort(key=lambda v: int(v["id"]))
    anns = [a for a in src.get("annotations", []) if int(a.get("image_id", -1)) in image_ids]
    out_obj = dict(src)
    out_obj["videos"] = videos
    out_obj["images"] = []
    counters = {}
    for image in images:
        q = dict(image)
        v = int(q["video_id"])
        q["frame_id"] = counters.get(v, 0)
        counters[v] = q["frame_id"] + 1
        out_obj["images"].append(q)
    out_obj["annotations"] = anns
    out_obj["tracks"] = [t for t in src.get("tracks", []) if int(t.get("video_id", -1)) in set(wanted)]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(out_obj))
    os.replace(tmp, out)
    return {"annotation_source": str(annotation.resolve()), "output": str(out.resolve()),
            "videos": len(videos), "video_ids": wanted, "images": len(images),
            "annotations": len(anns), "max_frames": max_frames,
            "bank_videos": len(roles["known_bank_train"]),
            "calibration_videos": len(roles["known_calibration"]),
            "audit_videos": len(roles["known_audit"]),
            "gt_labels_for_detector": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", default="data/iclr27_phase15s/sources/tao_train_annotations.json")
    ap.add_argument("--roles", default="outputs/iclr27_phase15s/manifests/data_split_and_leakage_audit.json")
    # Include ``validation`` in the filename: the frozen OVTR TAO loader
    # selects its TAO parser from this registered path token.
    ap.add_argument("--out", default="data/iclr27_phase15s/sources/validation_public_roles.json")
    ap.add_argument("--max-frames", type=int, default=12000)
    args = ap.parse_args()
    print(json.dumps(build(ROOT / args.annotation, ROOT / args.roles, ROOT / args.out, args.max_frames), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
