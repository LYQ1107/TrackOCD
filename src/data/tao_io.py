"""Helpers for reading TAO / TAO-OW annotations without modifying raw data."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TAO = PROJECT_ROOT / "data" / "raw" / "tao"
ANNOT_ROOT = RAW_TAO / "annotations"
FRAMES_ROOT = RAW_TAO / "frames"


def load_tao_annotations(split: str) -> dict:
    """Load one of train/validation/test TAO annotation json files."""
    if split == "val":
        split = "validation"
    path = ANNOT_ROOT / f"{split}.json"
    with open(path) as f:
        return json.load(f)


def load_coco2tao_map() -> dict:
    p = PROJECT_ROOT / "third_party" / "Open-World-Tracking" / "datasets" / "coco_id2tao_id.json"
    with open(p) as f:
        return {int(k): int(v) for k, v in json.load(f).items()}


def load_distractor_map() -> dict:
    p = PROJECT_ROOT / "third_party" / "Open-World-Tracking" / "datasets" / "distractor_classes.json"
    with open(p) as f:
        raw = json.load(f)
    return {int(k): [int(v) for v in vals] for k, vals in raw.items()}


def category_sets():
    """Return (known_ids, distractor_ids) from the official OW-Tracking maps."""
    coco2tao = load_coco2tao_map()
    dist_map = load_distractor_map()
    known = set(coco2tao.values())
    distractor = set()
    for vals in dist_map.values():
        distractor.update(vals)
    return known, distractor


def group_tracks(ann_data: dict, exclude_distractor=True):
    """Group annotations into tracks.

    Returns dict track_key -> track dict:
      sample_id, video_id, track_id, category_id, is_known, is_distractor,
      annotations (sorted by frame_index), first_frame, last_frame, num_frames.
    """
    known, distractor = category_sets()
    cat_ids = {c["id"] for c in ann_data["categories"]}
    img_by_id = {im["id"]: im for im in ann_data["images"]}

    anns_by_track = defaultdict(list)
    for ann in ann_data["annotations"]:
        vid = ann["video_id"]
        tid = ann["track_id"]
        cat = ann["category_id"]
        if cat not in cat_ids:
            raise ValueError(f"annotation {ann['id']} has unknown category_id {cat}")
        anns_by_track[(vid, tid)].append(ann)

    tracks = {}
    for (vid, tid), anns in anns_by_track.items():
        anns = sorted(anns, key=lambda a: img_by_id[a["image_id"]]["frame_index"])
        cat = anns[0]["category_id"]
        is_dist = cat in distractor
        if exclude_distractor and is_dist:
            continue
        tracks[(vid, tid)] = {
            "sample_id": f"{vid}_{tid}",
            "video_id": vid,
            "track_id": tid,
            "category_id": cat,
            "is_known": cat in known,
            "is_distractor": is_dist,
            "annotations": anns,
            "first_frame": img_by_id[anns[0]["image_id"]]["frame_index"],
            "num_frames": len(anns),
        }
    return tracks


def resolve_frame_path(image_rec: dict) -> Path:
    return FRAMES_ROOT / image_rec["file_name"]


def atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
