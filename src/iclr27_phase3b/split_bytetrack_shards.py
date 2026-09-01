"""Split the full detection JSONL files into balanced ByteTrack shards."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
SRC = ROOT / "outputs" / "iclr27_phase3b" / "full_export" / "pre_assoc_detections"
OUT_ROOT = ROOT / "outputs" / "iclr27_phase3b" / "bytetrack" / "shards"


def main():
    n = int(os.environ.get("PHASE3B_BT_SHARDS", "8"))
    manifest = json.load(open(ROOT / "outputs/iclr27_phase3b/frozen_detections/manifest.json"))
    items = []
    for p in sorted(SRC.glob("*.jsonl")):
        vid = int(p.stem)
        items.append((vid, p, manifest.get(str(vid), {}).get("detection_count", 0)))
    items.sort(key=lambda x: -x[2])
    buckets = [[] for _ in range(n)]
    loads = [0] * n
    for vid, p, cnt in items:
        i = min(range(n), key=lambda j: loads[j])
        buckets[i].append((vid, p))
        loads[i] += cnt
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for i, bucket in enumerate(buckets):
        d = OUT_ROOT / f"shard_{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for vid, p in bucket:
            (d / p.name).symlink_to(p)
        print(i, len(bucket), loads[i], d)
    print("total", sum(loads))


if __name__ == "__main__":
    main()
