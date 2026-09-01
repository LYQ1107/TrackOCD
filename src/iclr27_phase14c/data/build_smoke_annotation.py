"""Create a label-free OVTR smoke annotation for two disjoint calibration videos."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="0,51")
    ap.add_argument("--out", default="data/iclr27_phase14c/manifests/phase14c_smoke_validation.json")
    args = ap.parse_args()
    vids = {int(x) for x in args.videos.split(",") if x}
    src = json.loads((ROOT / "data/iclr27_phase14c/sources/tao_train_annotations.json").read_text())
    ims = defaultdict(list)
    for im in src["images"]:
        if int(im["video_id"]) in vids:
            ims[int(im["video_id"])].append(im)
    for v in ims:
        ims[v].sort(key=lambda x: (int(x.get("frame_index", 0)), int(x["id"])))
    videos = {int(v["id"]): v for v in src["videos"]}
    out = {
        "info": src.get("info", {}), "licenses": src.get("licenses", []),
        "categories": src.get("categories", []),
        "videos": [videos[v] for v in sorted(vids) if v in videos],
        "images": [], "tracks": [], "annotations": [],
    }
    for v in sorted(vids):
        for frame_id, im in enumerate(ims.get(v, [])):
            x = dict(im); x["frame_id"] = frame_id; out["images"].append(x)
    p = ROOT / args.out; p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(json.dumps(out, indent=2, sort_keys=True)); tmp.replace(p)
    print(json.dumps({"videos": sorted(vids), "images": len(out["images"]), "annotations": 0, "tracks": 0}))


if __name__ == "__main__":
    main()
