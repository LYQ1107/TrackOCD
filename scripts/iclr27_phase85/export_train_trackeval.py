#!/usr/bin/env python3
"""Export a replay and TRAIN annotations to a class-agnostic TrackEval TAO set.

The native Phase82R lineage intentionally omits category labels.  For a physical
MOT diagnostic we collapse TRAIN GT and predictions to one ``object`` category,
without feeding labels to inference.  The original TAO files are never changed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase85"
TRAIN_GT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/train.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gt", type=Path, default=TRAIN_GT)
    args = ap.parse_args()

    replay = [json.loads(line) for line in args.replay.read_text(encoding="utf-8").splitlines() if line.strip()]
    video_ids = {int(r["video_id"]) for r in replay}
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    videos = [dict(v) for v in gt["videos"] if int(v["id"]) in video_ids]
    kept_images = {int(im["id"]) for im in gt["images"] if int(im["video_id"]) in video_ids}
    images = [dict(im) for im in gt["images"] if int(im["id"]) in kept_images]
    annotations = [dict(a) for a in gt["annotations"] if int(a["image_id"]) in kept_images]
    tracks = [dict(t) for t in gt.get("tracks", []) if int(t.get("video_id", -1)) in video_ids]
    for v in videos:
        v["neg_category_ids"] = []
        v["not_exhaustive_category_ids"] = []
    for a in annotations:
        a["category_id"] = 1
    for t in tracks:
        t["category_id"] = 1
    gt_out = {
        "videos": videos,
        "images": images,
        "annotations": annotations,
        "tracks": tracks,
        "categories": [{"id": 1, "name": "object"}],
        "info": dict(gt.get("info", {})),
        "licenses": list(gt.get("licenses", [])),
    }
    predictions = []
    for r in replay:
        box = r.get("bbox_xyxy")
        if box is None:
            continue
        x0, y0, x1, y1 = [float(x) for x in box]
        predictions.append({
            "image_id": int(r["image_id"]),
            "bbox": [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)],
            "score": float(r.get("base_score", r.get("score", 0.0))),
            "category_id": 1,
            "video_id": int(r["video_id"]),
            "track_id": int(r["physical_track_id"]),
        })
    base = OUT / "trackeval" / args.tag
    gt_path = base / "gt" / "train_classagnostic.json"
    tracker_path = base / "trackers" / args.tag / "data" / "tao_track.json"
    atomic_json(gt_path, gt_out)
    atomic_json(tracker_path, predictions)
    manifest = {
        "schema_version": "trackocd.phase82r.classagnostic_trackeval_export.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tag": args.tag,
        "source_replay": str(args.replay),
        "source_replay_sha256": sha256(args.replay),
        "source_train_gt": str(args.gt),
        "source_train_gt_sha256": sha256(args.gt),
        "video_count": len(videos),
        "gt_image_count": len(images),
        "gt_annotation_count": len(annotations),
        "gt_track_count": len(tracks),
        "prediction_count": len(predictions),
        "gt_path": str(gt_path),
        "tracker_path": str(tracker_path),
        "gt_sha256": sha256(gt_path),
        "tracker_sha256": sha256(tracker_path),
        "class_agnostic_category": 1,
        "labels_used_for_inference": False,
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_as_model_input": False,
        "original_train_gt_unchanged": True,
    }
    atomic_json(base / "export_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
