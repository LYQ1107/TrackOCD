#!/usr/bin/env python3
"""TAO Validation image/frame adapters for the PANDAS pipeline."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


FRAME_RE = re.compile(r"(\d+)$")


def load_tao(annotation_path: Path) -> dict[str, Any]:
    with annotation_path.open() as handle:
        return json.load(handle)


def annotated_image_records(annotation_path: Path, frame_root: Path) -> list[dict[str, Any]]:
    """Return sorted image-only records; no annotation fields are retained."""
    data = load_tao(annotation_path)
    records = []
    for image in data["images"]:
        relative = Path(image["file_name"])
        path = frame_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({
            "image_id": int(image["id"]),
            "video_id": int(image["video_id"]),
            "frame_index": int(image["frame_index"]),
            "path": str(path),
            "width": int(image["width"]),
            "height": int(image["height"]),
        })
    return sorted(records, key=lambda item: (item["video_id"], item["frame_index"], item["image_id"]))


def full_video_frame_records(annotation_path: Path, frame_root: Path) -> dict[int, list[dict[str, Any]]]:
    """Enumerate all jpg frames for each of the 988 annotated Val videos."""
    data = load_tao(annotation_path)
    # Full-stream inference also includes unannotated frames.  Keep the
    # canonical image id when a frame is present in TAO's image table and use
    # a sentinel otherwise; export_bytetrack_trackeval later re-joins frames
    # by path and exports only canonical annotated image ids.
    image_id_by_path = {
        str((frame_root / Path(image["file_name"])).resolve()): int(image["id"])
        for image in data["images"]
    }
    result: dict[int, list[dict[str, Any]]] = {}
    for video in data["videos"]:
        relative = Path(video["name"]).relative_to("val")
        video_dir = frame_root / "val" / relative
        if not video_dir.is_dir():
            raise FileNotFoundError(video_dir)
        paths = sorted(
            video_dir.glob("*.jpg"),
            key=lambda path: (
                path.stem[: -len(FRAME_RE.search(path.stem).group(1))]
                if FRAME_RE.search(path.stem) else path.stem,
                int(FRAME_RE.search(path.stem).group(1))
                if FRAME_RE.search(path.stem) else -1,
            ),
        )
        records = []
        for ordinal, path in enumerate(paths):
            records.append({
                "video_id": int(video["id"]),
                "image_id": image_id_by_path.get(str(path.resolve()), -1),
                # Association only needs a deterministic consecutive stream;
                # annotated frames are joined back to their canonical TAO
                # image IDs by path before evaluation.
                "frame_index": ordinal,
                "path": str(path),
                "width": int(video["width"]),
                "height": int(video["height"]),
            })
        if not records:
            raise ValueError(f"no frames found for TAO Validation video {video['id']}")
        result[int(video["id"])] = records
    return result


def annotation_index(annotation_path: Path) -> dict[int, list[dict[str, Any]]]:
    data = load_tao(annotation_path)
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        result[int(annotation["image_id"])].append(annotation)
    return result


class TaoImageDataset(Dataset):
    """PANDAS-compatible TAO image dataset.

    ``include_ground_truth=False`` is the only mode used for discovery and
    inference.  In that mode the target contains empty tensors and no TAO
    category, track, or box data.
    """

    def __init__(self, records, transform, include_ground_truth=False, annotations=None):
        self.records = list(records)
        self.transform = transform
        self.include_ground_truth = include_ground_truth
        self.annotations = annotations or {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        if self.include_ground_truth:
            anns = self.annotations.get(record["image_id"], [])
            boxes = []
            labels = []
            areas = []
            iscrowd = []
            for annotation in anns:
                x, y, width, height = annotation["bbox"]
                boxes.append([x, y, x + width, y + height])
                labels.append(int(annotation["category_id"]))
                areas.append(float(annotation.get("area", width * height)))
                iscrowd.append(int(annotation.get("iscrowd", 0)))
        else:
            boxes, labels, areas, iscrowd = [], [], [], []
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([record["image_id"]], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            "height": record["height"],
            "width": record["width"],
            "size": record["width"] * record["height"],
        }
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target
