"""Build frozen detection manifest for the full 988-video stream."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
EXP = ROOT / "outputs" / "iclr27_phase3b" / "full_export"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    det_dir = EXP / "pre_assoc_detections"
    all_json = json.load(open(ROOT / "third_party/SimOWT/datasets/tao/annotations/val_split/all.json"))
    frame_count = defaultdict(int)
    for im in all_json["images"]:
        frame_count[im["video_id"]] += 1
    manifest = {}
    rows = []
    for p in sorted(det_dir.glob("*.jsonl")):
        vid = int(p.stem)
        scores = []
        boxes = []
        counts = defaultdict(int)
        n_det = 0
        for line in p.read_text().splitlines():
            r = json.loads(line)
            n_det += 1
            scores.append(r["score"])
            boxes.append(tuple(r["bbox_xyxy_original"]))
            counts[r["frame_order"]] += 1
        scores = np.asarray(scores, dtype=np.float64)
        bbox_checksum = hashlib.sha256(repr(sorted(boxes)).encode()).hexdigest()[:16]
        manifest[str(vid)] = {
            "video_id": vid,
            "frame_count": frame_count.get(vid, 0),
            "detection_count": n_det,
            "score_min": float(scores.min()) if scores.size else None,
            "score_max": float(scores.max()) if scores.size else None,
            "score_mean": float(scores.mean()) if scores.size else None,
            "score_p10": float(np.percentile(scores, 10)) if scores.size else None,
            "score_p50": float(np.percentile(scores, 50)) if scores.size else None,
            "score_p90": float(np.percentile(scores, 90)) if scores.size else None,
            "empty_frame_ratio": 1.0 - len(counts) / max(frame_count.get(vid, 1), 1),
            "bbox_checksum": bbox_checksum,
            "file_sha256": sha256(p),
        }
        rows.append(manifest[str(vid)])
    out_dir = ROOT / "outputs" / "iclr27_phase3b" / "frozen_detections"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    with open(out_dir / "file_hashes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    total_det = sum(r["detection_count"] for r in rows)
    print("videos", len(rows), "frames", sum(r["frame_count"] for r in rows),
          "detections", total_det)


if __name__ == "__main__":
    main()
