"""Build the smallest legal public TRAIN subset for a DSCT proposal audit.

Only public TRAIN videos are selected.  The output is an atomic annotation
subset; it contains no DEV+/Q1 labels and is used only for model inference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/iclr27_phase15r/sources/dsct_train_subset.json")
    ap.add_argument("--videos", nargs="+", type=int,
                    # Greedy public-TRAIN cover of every representation-train
                    # supported-known category (single-video categories remain
                    # single-video by construction).
                    default=[119, 569, 1086, 1277, 1836, 1955, 2010, 2031,
                             2049, 2074, 2161, 2170, 2300, 2362, 2460, 2667,
                             2722, 2977])
    args = ap.parse_args()
    src = json.loads((ROOT / "data/iclr27_phase15/sources/tao_train_annotations.json").read_text())
    wanted = set(args.videos)
    vids = [v for v in src["videos"] if int(v["id"]) in wanted]
    image_ids = {int(i["id"]) for i in src["images"] if int(i["video_id"]) in wanted}
    anns = [a for a in src["annotations"] if int(a.get("image_id", -1)) in image_ids]
    out = dict(src)
    out["videos"] = vids
    selected_images = [i for i in src["images"] if int(i["id"]) in image_ids]
    selected_images.sort(key=lambda i: (int(i["video_id"]), int(i.get("frame_index", 0)), int(i["id"])))
    frame_counter = {}
    out["images"] = []
    for image in selected_images:
        if int(image["id"]) not in image_ids:
            continue
        image = dict(image)
        # CocoVID used by Phase-6B expects frame_id; TAO TRAIN stores the same
        # zero-based value as frame_index.
        vid = int(image["video_id"])
        image["frame_id"] = frame_counter.get(vid, 0)
        frame_counter[vid] = image["frame_id"] + 1
        out["images"].append(image)
    out["annotations"] = anns
    # The original TAO JSON carries auxiliary track metadata; retain only
    # entries whose video belongs to the subset when the field is present.
    if isinstance(src.get("tracks"), list):
        out["tracks"] = [t for t in src["tracks"] if int(t.get("video_id", -1)) in wanted]
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, path)
    print(json.dumps({"videos": sorted(wanted), "n_videos": len(vids), "n_images": len(out["images"]), "n_annotations": len(anns), "out": str(path)}, indent=2))


if __name__ == "__main__":
    main()
