"""Split the YTVIS-format TAO train json into K chunks (whole videos per
chunk) so IDOL inference can run in parallel across GPUs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
SRC = ROOT / "third_party" / "SimOWT" / "datasets" / "tao" / "annotations" \
    / "train_agn_ytvis.json"
OUT_DIR = SRC.parent
K = 8


def main():
    d = json.load(open(SRC))
    vids = d["videos"]
    # Preserve original video order; chunk by video id blocks.
    chunks = [[] for _ in range(K)]
    for i, v in enumerate(vids):
        chunks[i % K].append(v["id"])
    vid_set = {v["id"]: v for v in vids}
    img_by_vid = {}
    for im in d["images"]:
        img_by_vid.setdefault(im["video_id"], []).append(im)
    ann_by_vid = {}
    for a in d["annotations"]:
        ann_by_vid.setdefault(a["video_id"], []).append(a)
    for k, ids in enumerate(chunks):
        ids = sorted(ids)
        out = {
            "info": d.get("info", {}), "licenses": d.get("licenses", []),
            "categories": d["categories"],
            "videos": [vid_set[i] for i in ids],
            "images": [im for i in ids for im in img_by_vid[i]],
            "annotations": [a for i in ids for a in ann_by_vid[i]],
        }
        p = OUT_DIR / f"train_agn_ytvis_chunk{k}.json"
        with open(p, "w") as f:
            json.dump(out, f)
        print(p, len(out["videos"]), len(out["images"]),
              len(out["annotations"]))


if __name__ == "__main__":
    main()
