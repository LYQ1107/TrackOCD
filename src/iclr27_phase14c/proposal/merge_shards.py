"""Merge completed physical proposal shard CSVs without ID collisions."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/iclr27_phase14c/proposals/proposals_mixed.csv")
    args = ap.parse_args()
    rows = []
    fieldnames = None
    seen = set()
    for rel in args.shards:
        with (ROOT / rel).open() as f:
            r = csv.DictReader(f)
            fieldnames = fieldnames or list(r.fieldnames)
            assert list(r.fieldnames) == fieldnames
            for x in r:
                key = (int(x["video_id"]), int(x["frame_id"]), int(x["proposal_local_id"]), int(x["track_id"]))
                assert key not in seen, f"duplicate physical row {key}"
                seen.add(key); rows.append(x)
    rows.sort(key=lambda x: (int(x["video_id"]), int(x["frame_id"]), int(x["proposal_local_id"]), int(x["track_id"])))
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    os.replace(tmp, out)
    print({"rows": len(rows), "tracks": len({(int(x['video_id']), int(x['track_id'])) for x in rows}), "videos": len({int(x['video_id']) for x in rows}), "out": str(out)})


if __name__ == "__main__":
    main()
