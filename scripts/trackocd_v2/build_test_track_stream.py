#!/usr/bin/env python3
"""Build the TAO Test physical manifest with an explicit semantic lock.

Before ``FINAL_FREEZE`` this command emits only a structural physical-track
manifest and source counts/hashes.  The final labelled manifest and evaluator
sidecar can be emitted only with ``--final`` after the immutable freeze file
exists.  No Test category names, IDs, or roles are written in structural mode.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import atomic_json, atomic_write_text, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.protocol import assert_test_semantic_access_allowed, load_ids, validate_roles  # noqa: E402


TEST_ANNOTATION = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_test_lvis_v1_classes.json")
FINAL_FREEZE = Path("/data2/usr_for_deadline/trackocd_v2/project_outputs/audit/FINAL_FREEZE.json")
KNOWN_IDS = ROOT / "data/tao_ow_ocd_v1/splits/known_ids.json"
NOVEL_IDS = ROOT / "data/tao_ow_ocd_v1/splits/unknown_ids_val.json"
DISTRACTOR_IDS = ROOT / "data/tao_ow_ocd_v1/splits/distractor_ids.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load() -> dict[str, Any]:
    value = json.loads(TEST_ANNOTATION.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("TAO Test annotation is not an object")
    return value


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = [float(value) for value in box]
    return [x, y, x + width, y + height]


def _records(
    annotation: dict[str, Any],
    *,
    include_categories: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build physical rows, optionally reading Test category assignments.

    The structural pre-freeze path must not inspect category IDs at all.  The
    flag is intentionally explicit so a future caller cannot accidentally
    turn a structural audit into semantic Test access by reusing this helper.
    """

    images = {int(row["id"]): row for row in annotation["images"]}
    grouped: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    categories: dict[tuple[int, int], set[int]] = defaultdict(set)
    for ann in annotation["annotations"]:
        key = (int(ann["video_id"]), int(ann["track_id"]))
        grouped[key].append(ann)
        if include_categories:
            categories[key].add(int(ann["category_id"]))

    rows: list[dict[str, Any]] = []
    category_by_key: dict[str, int] = {}
    for (video_id, track_id), anns in grouped.items():
        if include_categories and len(categories[(video_id, track_id)]) != 1:
            raise ValueError(f"Test track has multiple categories: {(video_id, track_id)}")
        anns = sorted(anns, key=lambda item: (int(images[int(item["image_id"])].get("frame_index", item["image_id"])), int(item["image_id"])))
        key = f"{video_id}_{track_id}"
        rows.append({
            "sample_key": key,
            "video_id": video_id,
            "physical_track_id": str(track_id),
            "frame_ids": [int(item["image_id"]) for item in anns],
            "boxes_xyxy": [_xywh_to_xyxy(item["bbox"]) for item in anns],
            "image_paths": [str(images[int(item["image_id"])]["file_name"]) for item in anns],
            "source_split": "test",
        })
        if include_categories:
            category_by_key[key] = next(iter(categories[(video_id, track_id)]))
    rows.sort(key=lambda row: (int(row["video_id"]), int(row["frame_ids"][0]), str(row["physical_track_id"])))
    for index, row in enumerate(rows):
        row["stream_order"] = index
    return rows, category_by_key


def _orders(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {"main": rows}
    for seed in (1027, 1028, 1029):
        videos = sorted({int(row["video_id"]) for row in rows})
        random.Random(seed).shuffle(videos)
        rank = {video: index for index, video in enumerate(videos)}
        ordered = sorted(rows, key=lambda row: (rank[int(row["video_id"])], int(row["frame_ids"][0]), str(row["physical_track_id"])))
        result[f"seed{seed}"] = [dict(row, stream_order=index) for index, row in enumerate(ordered)]
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def build(*, final: bool) -> dict[str, Any]:
    if not TEST_ANNOTATION.is_file():
        raise FileNotFoundError(TEST_ANNOTATION)
    # This guard must precede both annotation parsing and any category-role
    # lookup on the final path.  Structural mode is allowed to read only the
    # physical lineage fields and never calls the semantic accessor.
    if final:
        assert_test_semantic_access_allowed(FINAL_FREEZE, "build final TAO Test semantic manifest")
    annotation = _load()
    rows, category_by_key = _records(annotation, include_categories=final)
    orders = _orders(rows)
    out = ensure_output_layout()
    manifest_root = out / "manifests"
    structural_path = manifest_root / "tao_test_gt_tracks_structural.jsonl"
    _write_jsonl(structural_path, rows)
    payload: dict[str, Any] = {
        "schema_version": "trackocd.v2.test_track_stream.v1",
        "status": "STRUCTURAL_ONLY" if not final else "FINAL_LABELLED_MANIFEST_READY",
        "generated_utc": _now(),
        "annotation": str(TEST_ANNOTATION.resolve()),
        "annotation_sha256": sha256_file(TEST_ANNOTATION),
        "structural_manifest": str(structural_path.resolve()),
        "structural_manifest_sha256": sha256_file(structural_path),
        "counts": {
            "videos": len(annotation.get("videos", [])),
            "images": len(annotation.get("images", [])),
            "annotations": len(annotation.get("annotations", [])),
            "tracks": len(rows),
        },
        "orders": {name: f"tao_test_gt_tracks_{name}.jsonl" for name in orders},
        "test_semantic_accessed": bool(final),
        "selection_allowed_before_final_freeze": False,
    }
    if not final:
        atomic_json(out / "audit/test_track_stream.json", payload)
        return payload

    payload["counts"]["categories_in_annotations"] = len(set(category_by_key.values()))
    known = load_ids(KNOWN_IDS)
    novel = load_ids(NOVEL_IDS)
    distractor = load_ids(DISTRACTOR_IDS)
    validate_roles(known, novel, distractor)
    labels = []
    role_counts: Counter[str] = Counter()
    for key, category in sorted(category_by_key.items()):
        if category in distractor:
            role = "distractor"
        elif category in known:
            role = "old"
        elif category in novel:
            role = "new"
        else:
            raise ValueError(f"Test category outside locked role sets: {category}")
        labels.append({"sample_key": key, "gt_category_id": int(category), "gt_split": role})
        role_counts[role] += 1
    label_path = manifest_root / "private_tao_test_gt_track_labels.jsonl"
    _write_jsonl(label_path, labels)
    for name, ordered in orders.items():
        target = manifest_root / f"tao_test_gt_tracks_{name}.jsonl"
        _write_jsonl(target, ordered)
        payload["orders"][name] = str(target.resolve())
    final_path = manifest_root / "tao_test_gt_tracks.jsonl"
    _write_jsonl(final_path, orders["main"])
    payload.update({
        "final_manifest": str(final_path.resolve()),
        "final_manifest_sha256": sha256_file(final_path),
        "label_sidecar": str(label_path.resolve()),
        "label_sidecar_sha256": sha256_file(label_path),
        "role_counts": dict(role_counts),
        "known_ids_source": str(KNOWN_IDS.resolve()),
        "novel_ids_source": str(NOVEL_IDS.resolve()),
        "distractor_ids_source": str(DISTRACTOR_IDS.resolve()),
        "final_freeze": str(FINAL_FREEZE.resolve()),
        "final_freeze_sha256": sha256_file(FINAL_FREEZE),
    })
    atomic_json(out / "audit/test_track_stream.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="emit labelled Test manifest; requires FINAL_FREEZE")
    args = parser.parse_args()
    result = build(final=args.final)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
