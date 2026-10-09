#!/usr/bin/env python3
"""Export compact full-stream ByteTrack NPZ files to canonical TAO JSON.

The NPZ stream is generated over every frame of each TAO Validation video.
Only frames that have a canonical TAO image id are exported to TrackEval;
the association itself has already seen the intervening unannotated frames.
No prototype/category information is used here: TAO-OW's dataset adapter is
class agnostic and receives category_id=1 solely as its required placeholder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tao-annotation", type=Path, required=True)
    parser.add_argument("--tao-frame-root", type=Path, required=True)
    parser.add_argument("--detections-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    # Import the same canonical frame enumeration used by full-stream
    # inference. This makes the ordinal stored in each NPZ unambiguous.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tao_data import full_video_frame_records, load_tao

    tao = load_tao(args.tao_annotation)
    full_records = full_video_frame_records(args.tao_annotation, args.tao_frame_root)
    image_by_path = {}
    image_video = {}
    for image in tao["images"]:
        path = (args.tao_frame_root / image["file_name"]).resolve()
        image_by_path[str(path)] = int(image["id"])
        image_video[int(image["id"])] = int(image["video_id"])

    files = sorted(args.detections_root.glob("video_*.npz"))
    expected = set(full_records)
    found = {int(path.stem.split("_")[-1]) for path in files}
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise RuntimeError(f"full-stream NPZ coverage mismatch; missing={missing[:10]}, extra={extra[:10]}")

    annotations = []
    exported_frames = 0
    exported_rows = 0
    for path in files:
        video_id = int(path.stem.split("_")[-1])
        records = full_records[video_id]
        with np.load(path) as data:
            frame_index = data["frame_index"]
            offsets = data["frame_offsets"]
            boxes = data["boxes"]
            scores = data["score"]
            track_ids = data["track_id"]
            if len(frame_index) != len(records) or len(offsets) != len(records) + 1:
                raise RuntimeError(f"frame/offset length mismatch in {path}")
            if not (len(boxes) == len(scores) == len(track_ids) == int(offsets[-1])):
                raise RuntimeError(f"row length mismatch in {path}")
            for frame_pos, record in enumerate(records):
                image_id = image_by_path.get(str(Path(record["path"]).resolve()))
                if image_id is None:
                    continue
                if image_video[image_id] != video_id:
                    raise RuntimeError(f"video mismatch for TAO image {image_id}")
                exported_frames += 1
                begin, end = int(offsets[frame_pos]), int(offsets[frame_pos + 1])
                for row in range(begin, end):
                    x1, y1, x2, y2 = [float(value) for value in boxes[row]]
                    width = max(0.0, x2 - x1)
                    height = max(0.0, y2 - y1)
                    annotations.append({
                        "image_id": int(image_id),
                        "video_id": video_id,
                        "category_id": 1,
                        "track_id": int(track_ids[row]),
                        "bbox": [x1, y1, width, height],
                        "score": float(scores[row]),
                    })
                    exported_rows += 1

    annotations.sort(key=lambda item: (item["video_id"], item["image_id"], item["track_id"]))
    tracker_dir = args.output_root / "bytetrack" / "data"
    tracker_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = tracker_dir / "pred.json"
    prediction_path.write_text(json.dumps(annotations, separators=(",", ":")) + "\n")

    # TrackEval's TAO_OW adapter requires exactly one JSON in its GT folder.
    # Reuse the original 44 MiB file through a symlink rather than copying it.
    gt_dir = args.output_root / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_link = gt_dir / "validation.json"
    resolved_gt = args.tao_annotation.resolve()
    if gt_link.exists() or gt_link.is_symlink():
        if not gt_link.is_symlink() or gt_link.resolve() != resolved_gt:
            raise RuntimeError(f"refusing to replace existing GT path {gt_link}")
    else:
        gt_link.symlink_to(resolved_gt)

    result = {
        "status": "PASS",
        "format": "canonical TAO-OW TrackEval JSON",
        "association_input": "full 988-video frame stream; bbox/score only",
        "semantic_mapping_used": False,
        "videos": len(files),
        "annotated_frames_exported": exported_frames,
        "track_rows_exported": exported_rows,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "gt_path": str(gt_link),
        "gt_is_symlink": True,
    }
    audit_path = args.output_root / "export_trackeval_complete.json"
    audit_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
