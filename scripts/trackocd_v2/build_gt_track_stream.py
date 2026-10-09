#!/usr/bin/env python3
"""Build public GT-track streams and evaluator-only label sidecars."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.protocol import load_ids, validate_roles  # noqa: E402


TRAIN = ROOT / "data/raw/tao/annotations/train.json"
VAL = ROOT / "data/raw/tao/annotations/validation.json"
ROLE_ROOT = ROOT / "data/tao_ow_ocd_v1/splits"
LEGACY_STREAM_ROOT = ROOT / "data/tao_ow_ocd_v1/public"


def _load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def _role(category: int, known: set[int], novel: set[int], distractor: set[int], *, allow_train_unassigned: bool) -> str:
    if category in known:
        return "old"
    if category in novel:
        return "new"
    if category in distractor:
        return "distractor"
    return "train_unassigned" if allow_train_unassigned else "unassigned"


def _legacy_order(filename: str = "val_gt_track_stream.jsonl") -> list[str]:
    path = LEGACY_STREAM_ROOT / filename
    if not path.exists():
        return []
    order = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                order.append(f"{int(row['video_id'])}_{int(row.get('track_id', row.get('physical_track_id')))}")
    return order


def _build_records(annotation: dict, known: set[int], novel: set[int], distractor: set[int], *, include_distractors: bool) -> tuple[list[dict], list[dict]]:
    images = {int(row["id"]): row for row in annotation["images"]}
    videos = {int(row["id"]): row for row in annotation["videos"]}
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    categories: dict[tuple[int, int], int] = {}
    for ann in annotation["annotations"]:
        video_id = int(ann["video_id"])
        track_id = int(ann["track_id"])
        key = (video_id, track_id)
        grouped[key].append(ann)
        categories.setdefault(key, int(ann["category_id"]))

    records = []
    labels = []
    for (video_id, track_id), anns in grouped.items():
        category = categories[(video_id, track_id)]
        source_split = "train" if str(videos[video_id].get("name", "")).startswith("train/") else "val"
        role = _role(category, known, novel, distractor, allow_train_unassigned=source_split == "train")
        if role == "distractor" and not include_distractors:
            continue
        if role == "unassigned":
            raise ValueError(f"track category is outside locked role sets: {(video_id, track_id, category)}")
        anns = sorted(anns, key=lambda row: (int(images[int(row["image_id"])].get("frame_index", row["image_id"])), int(row["image_id"])))
        frame_ids = [int(row["image_id"]) for row in anns]
        boxes = [_xywh_to_xyxy(row["bbox"]) for row in anns]
        quality = []
        image_paths = []
        for row, box in zip(anns, boxes):
            image = images[int(row["image_id"])]
            width = max(float(image["width"]), 1.0)
            height = max(float(image["height"]), 1.0)
            quality.append(max((box[2] - box[0]) * (box[3] - box[1]) / (width * height), 1e-6))
            image_paths.append(str(image["file_name"]))
        sample_key = f"{video_id}_{track_id}"
        records.append({
            "sample_key": sample_key,
            "video_id": video_id,
            "physical_track_id": str(track_id),
            "frame_ids": frame_ids,
            "boxes_xyxy": boxes,
            "image_paths": image_paths,
            "quality": quality,
            "source_split": source_split,
        })
        labels.append({"sample_key": sample_key, "video_id": video_id, "physical_track_id": str(track_id), "gt_category_id": category, "gt_split": role})

    legacy = _legacy_order() if annotation is not None and records and records[0].get("source_split") == "val" else []
    rank = {key: index for index, key in enumerate(legacy)}
    records.sort(key=lambda row: (rank.get(row["sample_key"], len(rank) + int(row["video_id"])), int(row["video_id"]), int(row["physical_track_id"])))
    label_by_key = {row["sample_key"]: row for row in labels}
    labels = [label_by_key[row["sample_key"]] for row in records]
    return records, labels


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-distractors", action="store_true")
    args = parser.parse_args()
    known = load_ids(ROLE_ROOT / "known_ids.json")
    novel = load_ids(ROLE_ROOT / "unknown_ids_val.json")
    distractor = load_ids(ROLE_ROOT / "distractor_ids.json")
    validate_roles(known, novel, distractor)
    out = ensure_output_layout()
    train_records, train_labels = _build_records(_load(TRAIN), known, novel, distractor, include_distractors=args.include_distractors)
    val_records, val_labels = _build_records(_load(VAL), known, novel, distractor, include_distractors=args.include_distractors)
    manifests = out / "manifests"
    _write_jsonl(manifests / "tao_train_gt_tracks.jsonl", train_records)
    _write_jsonl(manifests / "tao_val_gt_tracks.jsonl", val_records)
    _write_jsonl(manifests / "private_tao_train_gt_track_labels.jsonl", train_labels)
    _write_jsonl(manifests / "private_tao_val_gt_track_labels.jsonl", val_labels)

    orders = {"main": val_records}
    for seed in (1027, 1028, 1029):
        legacy = _legacy_order(f"val_gt_track_stream_seed{seed}.jsonl")
        by_key = {row["sample_key"]: row for row in val_records}
        if set(legacy) != set(by_key):
            raise ValueError(f"legacy stream order mismatch for seed {seed}: {len(legacy)} vs {len(by_key)}")
        orders[f"seed{seed}"] = [by_key[key] for key in legacy]
        _write_jsonl(manifests / f"tao_val_gt_tracks_seed{seed}.jsonl", orders[f"seed{seed}"])
    order_meta = {
        "schema_version": "trackocd.v2.gt_track_stream.v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "public_fields_exclude": ["gt_category_id", "gt_split"],
        "label_sidecars": {
            "train": str((manifests / "private_tao_train_gt_track_labels.jsonl").resolve()),
            "val": str((manifests / "private_tao_val_gt_track_labels.jsonl").resolve()),
        },
        "stream_order": {name: "reused tao_ow_ocd_v1/public ordering" for name in orders},
        "counts": dict({name: len(rows) for name, rows in orders.items()}, train=len(train_records)),
        "include_distractors": bool(args.include_distractors),
        "train_annotation_sha256": sha256_file(TRAIN),
        "val_annotation_sha256": sha256_file(VAL),
    }
    atomic_json(out / "audit/gt_track_stream.json", order_meta)
    print(json.dumps(order_meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
