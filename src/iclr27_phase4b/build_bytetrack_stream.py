"""Build a TrackOCD-format track stream from ByteTrack predictions."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
PRED_DIR = ROOT / "outputs" / "iclr27_phase3b" / "bytetrack" / "predictions"
ALL_JSON = ROOT / "third_party/SimOWT/datasets/tao/annotations/val_split/all.json"
OUT = ROOT / "outputs" / "iclr27_phase4b" / "bytetrack" / "pred_track_stream_bytetrack.jsonl"


def main():
    all_data = json.load(open(ALL_JSON))
    img_meta = {im["id"]: im for im in all_data["images"]}
    tracks = defaultdict(lambda: {"frames": [], "boxes": [], "scores": []})
    for p in sorted(PRED_DIR.glob("*.json")):
        recs = json.loads(p.read_text())
        for r in recs:
            vid, tid = r["video_id"], r["track_id"]
            im = img_meta[r["image_id"]]
            x, y, w, h = r["bbox"]
            tracks[(vid, tid)]["frames"].append(r["image_id"])
            tracks[(vid, tid)]["boxes"].append([x, y, x + w, y + h])
            tracks[(vid, tid)]["scores"].append(r["score"])
    rows = []
    for (vid, tid), t in tracks.items():
        order = sorted(range(len(t["frames"])), key=lambda i: (img_meta[t["frames"][i]]["frame_index"], t["frames"][i]))
        frames = [t["frames"][i] for i in order]
        boxes = [t["boxes"][i] for i in order]
        scores = [t["scores"][i] for i in order]
        rows.append({
            "sample_id": f"B{vid}_{tid}",
            "video_id": vid,
            "track_id": tid,
            "frame_ids": frames,
            "image_paths": [img_meta[f]["file_name"] for f in frames],
            "boxes_xyxy": boxes,
            "areas": [(b[2]-b[0])*(b[3]-b[1]) for b in boxes],
            "scores": scores,
            "stream_order": 0,
        })
    rows.sort(key=lambda r: (r["video_id"], r["frame_ids"][0]))
    for i, r in enumerate(rows):
        r["stream_order"] = i
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print("tracks", len(rows), "->", OUT)


if __name__ == "__main__":
    main()
