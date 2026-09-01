"""Convert official OVTR TAO tracking JSON to a physical-only CSV."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--annotation", default="data/iclr27_phase14c/manifests/phase14c_validation_train.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ann = json.loads((ROOT / args.annotation).read_text())
    imgs = {int(x["id"]): x for x in ann["images"]}
    data = json.loads(Path(args.results_json).read_text())
    if not isinstance(data, list):
        raise ValueError("OVTR track result must be a list")
    by_frame: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for raw in data:
        if int(raw.get("image_id", -1)) not in imgs:
            continue
        r = dict(raw)
        r["video_id"] = int(r.get("video_id", imgs[int(r["image_id"])] ["video_id"]))
        r["image_id"] = int(r["image_id"])
        by_frame[(r["video_id"], r["image_id"])].append(r)
    rows = []
    for (vid, iid), rs in sorted(by_frame.items(), key=lambda kv: (
            int(kv[0][0]), int(imgs[kv[0][1]].get("frame_id", 0)), int(kv[0][1]))):
        im = imgs[iid]
        rs.sort(key=lambda r: (int(r.get("track_id", -1)), -float(r.get("score", 0.0))))
        for local, r in enumerate(rs):
            b = [float(x) for x in r["bbox"][:4]]
            # TAO result bbox is xywh; keep an explicit xyxy field.
            x1, y1, w, h = b
            rows.append({
                "video_id": vid,
                "frame_id": int(im.get("frame_id", 0)),
                "source_frame_index": int(im.get("frame_index", im.get("frame_id", 0))),
                "image_id": iid,
                "proposal_local_id": local,
                "track_id": int(r.get("track_id", -1)),
                "score": float(r.get("score", 0.0)),
                "bbox_xyxy": json.dumps([x1, y1, x1 + w, y1 + h], separators=(",", ":")),
                "det_category_id": int(r.get("category_id", -1)),
                "source_family": str(im["file_name"].split("/")[1]),
            })
    rows.sort(key=lambda r: (r["video_id"], r["frame_id"], r["proposal_local_id"], r["track_id"]))
    by_track = defaultdict(list)
    for i, r in enumerate(rows):
        by_track[(r["video_id"], r["track_id"])].append(i)
    for key, idxs in by_track.items():
        seen = set()
        for i in sorted(idxs, key=lambda j: (rows[j]["frame_id"], rows[j]["proposal_local_id"])):
            rows[i]["prior_hits"] = len(seen)
            seen.add(rows[i]["frame_id"])
    fields = ["video_id", "frame_id", "source_frame_index", "image_id",
              "proposal_local_id", "track_id", "score", "bbox_xyxy",
              "det_category_id", "source_family", "prior_hits"]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    print(json.dumps({"rows": len(rows), "tracks": len(by_track), "videos": len({r["video_id"] for r in rows}), "out": str(out)}))


if __name__ == "__main__":
    main()
