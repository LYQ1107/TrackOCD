#!/usr/bin/env python3
"""Merge SimOWT per-image prediction JSONs into a TAO-format tracker JSON and
build the public predicted-track stream for feature extraction."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-json", default="outputs/simowt/val_predictions.json")
    ap.add_argument("--stream-jsonl", default="data/tao_ow_ocd_v1/public/pred_track_stream.jsonl")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    all_anns = []
    if input_dir.name == "runs":
        json_files = sorted(input_dir.glob("simowt_inference*.json"))
    else:
        json_files = sorted(input_dir.glob("*.json"))
    for p in json_files:
        all_anns.extend(json.loads(p.read_text()))
    print(f"read {len(json_files)} per-frame JSON files")
    # de-duplicate just in case
    seen = set()
    unique = []
    for a in all_anns:
        key = (a.get("image_id"), a.get("track_id"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(a)
    unique.sort(key=lambda a: (a.get("video_id", 0), a.get("image_id", 0), a.get("track_id", 0)))

    out_json = PROJECT_ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(unique, indent=1))
    print(f"merged {len(unique)} detections -> {out_json}")

    # build track stream
    tracks = defaultdict(list)
    for a in unique:
        tracks[(a["video_id"], a["track_id"])].append(a)
    img_to_frame = {}
    img_to_name = {}
    ann_data = json.load(
        open(PROJECT_ROOT / "data" / "raw" / "tao" / "annotations" / "validation.json")
    )
    for im in ann_data["images"]:
        img_to_frame[im["id"]] = im["frame_index"]
        img_to_name[im["id"]] = im["file_name"]

    rows = []
    for (vid, tid), anns in tracks.items():
        anns = sorted(anns, key=lambda a: img_to_frame.get(a["image_id"], 0))
        boxes = []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
        rows.append(
            {
                "sample_id": f"P{vid}_{tid}",
                "video_id": vid,
                "track_id": tid,
                "frame_ids": [a["image_id"] for a in anns],
                "image_paths": [
                    img_to_name[a["image_id"]]
                    for a in anns
                ],
                "boxes_xyxy": boxes,
                "areas": [b[2] * b[3] for b in boxes],
                "scores": [a["score"] for a in anns],
                "stream_order": 0,
            }
        )
    rows.sort(key=lambda r: (r["video_id"], r["stream_order"]))
    for i, r in enumerate(rows):
        r["stream_order"] = i
    stream_path = PROJECT_ROOT / args.stream_jsonl
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stream_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"pred track stream: {len(rows)} tracks -> {stream_path}")


if __name__ == "__main__":
    main()
