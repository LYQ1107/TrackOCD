#!/usr/bin/env python3
"""Register the frozen predicted physical-track stream without GT joins."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


SOURCE = ROOT / "data/tao_ow_ocd_v1/public/pred_track_stream.jsonl"
OUTPUT = OUTPUT_TARGET / "manifests/tao_val_predicted_tracks.jsonl"


def convert(row: dict) -> dict:
    sample_id = str(row["sample_id"])
    frame_ids = [int(value) for value in row["frame_ids"]]
    boxes = [[float(value) for value in box] for box in row["boxes_xyxy"]]
    image_paths = [str(value) for value in row["image_paths"]]
    scores = [float(value) for value in row.get("scores", [1.0] * len(frame_ids))]
    if not (len(frame_ids) == len(boxes) == len(image_paths) == len(scores)):
        raise ValueError("predicted track lineage lengths differ for %s" % sample_id)
    return {
        "sample_key": "pred_" + sample_id,
        "source_sample_id": sample_id,
        "video_id": int(row["video_id"]),
        "physical_track_id": str(row["track_id"]),
        "frame_ids": frame_ids,
        "boxes_xyxy": boxes,
        "image_paths": image_paths,
        "quality": [max(value, 1e-6) for value in scores],
        "scores": scores,
        "areas": [float(value) for value in row.get("areas", [0.0] * len(frame_ids))],
        "stream_order": int(row.get("stream_order", 0)),
        "source_split": "val_predicted",
    }


def main() -> int:
    out = ensure_output_layout()
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name("." + OUTPUT.name + ".tmp.%d" % os.getpid())
    count = 0
    videos = set()
    observations = 0
    with SOURCE.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = convert(json.loads(line))
            target.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
            videos.add(row["video_id"])
            observations += len(row["frame_ids"])
    os.replace(temporary, OUTPUT)
    audit = {
        "schema_version": "trackocd.v2.predicted_track_stream.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": str(SOURCE.resolve()),
        "source_sha256": sha256_file(SOURCE),
        "output": str(OUTPUT.resolve()),
        "tracks": count,
        "videos": len(videos),
        "observations": observations,
        "public_fields_exclude": ["gt_category_id", "gt_split", "gt_category_name", "gt_match_id"],
        "gt_matching_used_to_build_stream": False,
        "historical_matched_stream_not_used": str((ROOT / "data/tao_ow_ocd_v1/public/pred_track_stream_matched_iou0.5.jsonl").resolve()),
        "test_semantic_accessed": False,
    }
    atomic_json(out / "audit/predicted_track_stream.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
