"""Deterministically partition mixed evaluation videos by frame count."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", default="data/iclr27_phase14c/manifests/phase14c_validation_train.json")
    ap.add_argument("--videos", default="outputs/iclr27_phase14c/manifests/mixed_eval_split.json")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="outputs/iclr27_phase14c/manifests/shards.json")
    args = ap.parse_args()
    ann = json.loads((ROOT / args.annotation).read_text())
    pop = json.loads((ROOT / args.videos).read_text())
    vids = [int(v) for v in pop["selected_videos"]]
    counts = {v: 0 for v in vids}
    for im in ann["images"]:
        counts[int(im["video_id"])] += 1
    n = max(1, min(int(args.num_shards), len(vids), 4))
    shards = [{"video_ids": [], "frame_count": 0} for _ in range(n)]
    for vid in sorted(vids, key=lambda v: (-counts[v], v)):
        j = min(range(n), key=lambda i: (shards[i]["frame_count"], i))
        shards[j]["video_ids"].append(vid)
        shards[j]["frame_count"] += counts[vid]
    for s in shards:
        s["video_ids"].sort()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps({"num_shards": n, "shards": shards}, indent=2, sort_keys=True))
    tmp.replace(out)
    print(json.dumps({"num_shards": n, "shards": shards}, indent=2))


if __name__ == "__main__":
    main()
