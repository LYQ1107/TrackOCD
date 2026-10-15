#!/usr/bin/env python3
"""Build a fixed, anonymous COVTrack Train dense-frame smoke input.

Only the approved structural image plan and the raw Train frame directory are
read.  The TAO Train annotation JSON is deliberately not opened.  The output
contains an empty annotation list and only COVTrack's own public LVIS class
names, which are required by its dataset parser and prompt checkpoint; no TAO
category IDs, names, Novel vocabulary, or GT arrays are imported.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v12_images/private_exact_joint_proof_or_assignment.json"
DEFAULT_FRAME_ROOT = Path("/data3/liuyeqiang/TAO-Amodal/frames")
DEFAULT_FRAME_ALIAS = ROOT / "outputs/trackocd_v2_runtime/frontend_inputs/covtrack_train_dense_smoke_v1/val_frame_alias"
DEFAULT_CLASSES = ROOT / "outputs/trackocd_v2_runtime/frontend_inputs/covtrack_val/lvis_classes_v1.txt"
DEFAULT_ANNOTATION = ROOT / "outputs/trackocd_v2_runtime/frontend_inputs/covtrack_train_dense_smoke_v1/anonymous_train.json"
DEFAULT_AUDIT = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_smoke_v1/input_audit.json"
VIDEO_IDS = (0, 1)
FULL_SHARD_SIZES = (102, 102, 102, 101)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _frame_key(entry: os.DirEntry[str]) -> tuple[int, int | str]:
    match = re.fullmatch(r"frame(\d+)", Path(entry.name).stem)
    if match:
        return (0, int(match.group(1)))
    return (1, entry.name)


def _load_classes(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("the public COVTrack class source is empty or duplicated")
    return names


def build(*, plan_path: Path, frame_root: Path, frame_alias: Path, classes_path: Path, annotation_path: Path, audit_path: Path, video_ids: tuple[int, ...] = VIDEO_IDS, shard_index: int | None = None) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("not_model_facing") is not True:
        raise ValueError("the selected image plan is not marked not_model_facing")
    if plan.get("semantic_labels_or_GT_boxes_in_image_plan") is not False:
        raise ValueError("the selected image plan has a semantic/GT boundary violation")
    rows = plan.get("image_metadata")
    if not isinstance(rows, list):
        raise ValueError("image plan has no image_metadata list")
    by_video: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        video_id = int(row["video_id"])
        by_video.setdefault(video_id, []).append(row)
    if len(by_video) != 407:
        raise ValueError(f"expected the approved 407-video plan, found {len(by_video)} videos")
    approved_video_ids = sorted(by_video)
    approved_video_id_order_sha256 = hashlib.sha256(json.dumps(approved_video_ids, separators=(",", ":")).encode()).hexdigest()
    if shard_index is not None:
        if not 0 <= shard_index < len(FULL_SHARD_SIZES):
            raise ValueError("approved shard index must be in 0..3")
        start = sum(FULL_SHARD_SIZES[:shard_index])
        video_ids = tuple(approved_video_ids[start : start + FULL_SHARD_SIZES[shard_index]])
    if not video_ids or len(set(video_ids)) != len(video_ids) or any(video_id not in by_video for video_id in video_ids):
        raise ValueError("video_ids must be a non-empty unique subset of the approved 407 IDs")
    class_names = _load_classes(classes_path)
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(class_names)]
    frame_alias.parent.mkdir(parents=True, exist_ok=True)
    if frame_alias.is_symlink():
        if frame_alias.resolve() != frame_root.resolve():
            raise ValueError(f"existing frame alias points elsewhere: {frame_alias}")
    elif frame_alias.exists():
        raise FileExistsError(f"refusing to replace non-symlink frame alias: {frame_alias}")
    else:
        frame_alias.symlink_to(frame_root, target_is_directory=True)

    videos: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    per_video: dict[str, dict[str, Any]] = {}
    next_image_id = 1_000_000
    for video_id in video_ids:
        selected_rows = by_video.get(video_id)
        if not selected_rows:
            raise ValueError(f"fixed video {video_id} is absent from the approved plan")
        selected_rows = sorted(selected_rows, key=lambda row: int(row["frame_index"]))
        source_files = {str(row["file_name"]) for row in selected_rows}
        source_paths = [frame_root / name for name in source_files]
        source_dirs = {path.parent for path in source_paths}
        if len(source_dirs) != 1:
            raise ValueError(f"video {video_id} maps to multiple frame directories")
        source_dir = next(iter(source_dirs))
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        entries = [entry for entry in os.scandir(source_dir) if entry.is_file() and entry.name.lower().endswith(".jpg")]
        entries.sort(key=_frame_key)
        if not entries:
            raise ValueError(f"no JPEG frames found for video {video_id}: {source_dir}")
        names = [entry.name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate frame names for video {video_id}")
        numeric_indices = []
        for entry in entries:
            match = re.fullmatch(r"frame(\d+)", Path(entry.name).stem)
            if match is None:
                numeric_indices = []
                break
            numeric_indices.append(int(match.group(1)))
        if numeric_indices and numeric_indices != list(range(numeric_indices[0], numeric_indices[-1] + 1)):
            raise ValueError(f"numeric frame sequence has a gap for video {video_id}")
        metadata_by_frame = {int(row["frame_index"]): row for row in selected_rows}
        source_frame_offset: int | None = None
        if numeric_indices:
            offsets = []
            for row in selected_rows:
                match = re.fullmatch(r"frame(\d+)", Path(str(row["file_name"])).stem)
                if match is None:
                    raise ValueError(f"approved plan row is not numeric for video {video_id}")
                offsets.append(int(match.group(1)) - int(row["frame_index"]))
            if len(set(offsets)) != 1:
                raise ValueError(f"approved plan/file naming offset is inconsistent for video {video_id}")
            source_frame_offset = offsets[0]
            plan_indices = sorted(metadata_by_frame)
            expected_first = plan_indices[0] + source_frame_offset
            expected_last = plan_indices[-1] + source_frame_offset
            if expected_first != numeric_indices[0] or expected_last != numeric_indices[-1]:
                raise ValueError(f"approved plan span disagrees with raw frame span for video {video_id}")
        width_height = {(int(row["width"]), int(row["height"])) for row in selected_rows}
        if len(width_height) != 1:
            raise ValueError(f"video {video_id} has inconsistent approved dimensions")
        width, height = next(iter(width_height))
        videos.append({
            "id": video_id,
            "name": f"anonymous_train_video_{video_id:04d}",
            "width": width,
            "height": height,
        })
        for local_frame_id, entry in enumerate(entries):
            source_frame_index = numeric_indices[local_frame_id] if numeric_indices else local_frame_id
            relative_name = Path(entry.path).relative_to(frame_root).as_posix()
            images.append({
                "id": next_image_id,
                "video_id": video_id,
                "frame_id": local_frame_id,
                "frame_index": source_frame_index,
                "file_name": relative_name,
                "width": width,
                "height": height,
            })
            next_image_id += 1
        per_video[str(video_id)] = {
            "frame_directory": str(source_dir.resolve()),
            "frames": len(entries),
            "first_frame_name": names[0],
            "last_frame_name": names[-1],
            "numeric_sequence": bool(numeric_indices),
            "source_filename_minus_plan_frame_index": source_frame_offset,
            "all_files_present": all((source_dir / name).is_file() for name in names),
            "width": width,
            "height": height,
        }

    annotation = {
        "info": {
            "description": "Anonymous COVTrack Train dense-frame smoke input; no TAO GT annotations",
            "source_split": "train_predicted",
        },
        "videos": videos,
        "images": images,
        "annotations": [],
        "tracks": [],
        "categories": categories,
        "licenses": [],
    }
    _atomic_json(annotation_path, annotation)
    audit = {
        "schema_version": "trackocd.cvpr.covtrack_train_dense_smoke_input_audit.v1",
        "status": "PASS_FIXED_ANONYMOUS_TRAIN_DENSE_INPUT",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection": {
            "video_ids": list(video_ids),
            "rule": (
                f"sorted approved video-ID order shard {shard_index}/4 with fixed sizes {list(FULL_SHARD_SIZES)}"
                if shard_index is not None
                else "fixed predeclared video IDs from the approved 407-video image plan"
            ),
            "outcome_selection": False,
            "approved_video_id_order_sha256": approved_video_id_order_sha256,
            "shard_index": shard_index,
        },
        "sources": {
            "image_plan": str(plan_path.resolve()),
            "image_plan_sha256": _sha256(plan_path),
            "frame_root": str(frame_root.resolve()),
            "frame_alias": str(frame_alias),
            "frame_alias_target": str(frame_alias.resolve()),
            "public_frontend_vocabulary": str(classes_path.resolve()),
            "public_frontend_vocabulary_sha256": _sha256(classes_path),
            "anonymous_annotation": str(annotation_path.resolve()),
            "anonymous_annotation_sha256": _sha256(annotation_path),
        },
        "counts": {
            "videos": len(videos),
            "images": len(images),
            "annotations": 0,
            "tracks": 0,
            "public_vocabulary_categories": len(categories),
        },
        "per_video": per_video,
        "boundaries": {
            "tao_train_annotation_opened": False,
            "tao_val_accessed": False,
            "tao_test_accessed": False,
            "gt_or_semantic_arrays_read": False,
            "tao_category_values_read": False,
            "novel_vocabulary_read": False,
            "public_frontend_vocabulary_role": "COVTrack parser/checkpoint input only; not TAO GT or model-selection labels",
            "pixels_decoded_or_hashed": False,
            "compatibility_alias_is_symlink_only": True,
            "model_executed": False,
            "training_started": False,
            "full_407_video_input_built": False,
        },
        "interpretation": "Input construction prerequisite only; no OCD/MOT metric or frontend qualification is claimed.",
    }
    _atomic_json(audit_path, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--frame-root", type=Path, default=DEFAULT_FRAME_ROOT)
    parser.add_argument("--frame-alias", type=Path, default=DEFAULT_FRAME_ALIAS)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--video-ids", type=int, nargs="+", default=list(VIDEO_IDS))
    parser.add_argument("--video-id-min", type=int, default=None)
    parser.add_argument("--video-id-max", type=int, default=None)
    parser.add_argument("--approved-shard-index", type=int, choices=range(4), default=None)
    args = parser.parse_args()
    if args.approved_shard_index is not None and (args.video_id_min is not None or args.video_id_max is not None or args.video_ids != list(VIDEO_IDS)):
        parser.error("--approved-shard-index cannot be combined with explicit video selection")
    if (args.video_id_min is None) != (args.video_id_max is None):
        parser.error("--video-id-min and --video-id-max must be supplied together")
    selected_video_ids = (
        tuple(range(args.video_id_min, args.video_id_max + 1))
        if args.video_id_min is not None
        else tuple(args.video_ids)
    )
    result = build(
        plan_path=args.plan.resolve(),
        frame_root=args.frame_root.resolve(),
        frame_alias=args.frame_alias,
        classes_path=args.classes.resolve(),
        annotation_path=args.annotation.resolve(),
        audit_path=args.audit.resolve(),
        video_ids=selected_video_ids,
        shard_index=args.approved_shard_index,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
