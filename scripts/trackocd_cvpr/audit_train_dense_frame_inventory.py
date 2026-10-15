#!/usr/bin/env python3
"""Audit dense Train frame filename coverage without opening image bytes.

The approved image plan contains only structural image metadata.  This audit
uses that plan to identify the 407 allowed Train video directories, lists
direct JPG entries, and checks the ordering/contiguity contract.  It never
loads annotations, categories, boxes, labels, pixels, weights, Val, or Test.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/trackocd_cvpr/train_dense_frame_inventory_v1.json"
IMAGE_PLAN = ROOT / "outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v12_images/private_exact_joint_proof_or_assignment.json"
FRAME_ROOT = Path("/data3/liuyeqiang/TAO-Amodal/frames")
OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/train_dense_frame_inventory_v1/result.json"
FRAME_NAME = re.compile(r"^(?:frame)?(\d+)\.jpg$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def audit() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan = json.loads(IMAGE_PLAN.read_text(encoding="utf-8"))
    violations: list[str] = []
    if plan.get("not_model_facing") is not True:
        violations.append("image_plan_not_marked_not_model_facing")
    if plan.get("semantic_labels_or_GT_boxes_in_image_plan") is not False:
        violations.append("image_plan_semantic_boundary_changed")
    rows = plan.get("image_metadata")
    if not isinstance(rows, list):
        raise ValueError("approved image plan has no image_metadata list")

    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        video_id = int(row["video_id"])
        file_name = str(row["file_name"])
        relative = Path(file_name)
        if relative.is_absolute() or ".." in relative.parts or not file_name.startswith("train/"):
            violations.append(f"unsafe_train_file_name:{file_name}")
        by_video[video_id].append(row)

    expected_video_count = 407
    if len(by_video) != expected_video_count:
        violations.append(f"video_count:{len(by_video)}!={expected_video_count}")

    video_rows: list[dict[str, Any]] = []
    listing_digest = hashlib.sha256()
    for video_id, image_rows in sorted(by_video.items()):
        directories = {Path(str(row["file_name"])).parent for row in image_rows}
        if len(directories) != 1:
            violations.append(f"multiple_directories:{video_id}")
            continue
        relative_directory = next(iter(directories))
        directory = FRAME_ROOT / relative_directory
        if not directory.is_dir():
            violations.append(f"missing_directory:{video_id}:{relative_directory}")
            continue
        names = sorted(
            entry.name
            for entry in directory.iterdir()
            if entry.is_file() and entry.suffix.lower() == ".jpg"
        )
        name_set = set(names)
        expected_names = {Path(str(row["file_name"])).name for row in image_rows}
        missing_plan_names = sorted(expected_names - name_set)
        if missing_plan_names:
            violations.append(f"missing_approved_files:{video_id}:{len(missing_plan_names)}")

        position = {name: index for index, name in enumerate(names)}
        frame_indices = [int(row["frame_index"]) for row in image_rows]
        min_frame = min(frame_indices)
        max_frame = max(frame_indices)
        expected_span = max_frame - min_frame + 1
        span_count_matches = len(names) == expected_span
        if not span_count_matches:
            violations.append(f"frame_span_count:{video_id}:{len(names)}!={expected_span}")
        position_mismatches = 0
        for row in image_rows:
            name = Path(str(row["file_name"])).name
            if name not in position or position[name] != int(row["frame_index"]) - min_frame:
                position_mismatches += 1
        if position_mismatches:
            violations.append(f"frame_order_mismatch:{video_id}:{position_mismatches}")

        numeric_values: list[int] = []
        nonstandard_names = 0
        for name in names:
            match = FRAME_NAME.match(name)
            if match is None:
                nonstandard_names += 1
            else:
                numeric_values.append(int(match.group(1)))
        numeric_contiguous = None
        if numeric_values:
            numeric_contiguous = len(numeric_values) == max(numeric_values) - min(numeric_values) + 1
            if not numeric_contiguous:
                violations.append(f"numeric_gap:{video_id}")

        listing_digest.update(f"{video_id}\0{relative_directory}\0".encode())
        for name in names:
            listing_digest.update(name.encode())
            listing_digest.update(b"\0")
        dataset = relative_directory.parts[1] if len(relative_directory.parts) > 1 else "unknown"
        video_rows.append(
            {
                "video_id": video_id,
                "dataset": dataset,
                "directory": str(relative_directory),
                "approved_image_rows": len(image_rows),
                "jpg_files": len(names),
                "min_approved_frame_index": min_frame,
                "max_approved_frame_index": max_frame,
                "frame_span_count_matches": span_count_matches,
                "approved_file_names_present": not missing_plan_names,
                "frame_order_matches_approved_offsets": position_mismatches == 0,
                "numeric_file_names_contiguous": numeric_contiguous,
                "nonstandard_jpg_names": nonstandard_names,
            }
        )

    dataset_summary: dict[str, dict[str, int]] = {}
    for dataset in sorted({row["dataset"] for row in video_rows}):
        group = [row for row in video_rows if row["dataset"] == dataset]
        dataset_summary[dataset] = {
            "videos": len(group),
            "jpg_files": sum(row["jpg_files"] for row in group),
            "approved_image_rows": sum(row["approved_image_rows"] for row in group),
            "order_pass": sum(row["frame_order_matches_approved_offsets"] for row in group),
        }

    status = "PASS_TRAIN_DENSE_FRAME_METADATA_407" if not violations else "FAIL_TRAIN_DENSE_FRAME_METADATA_407"
    return {
        "schema_version": "trackocd.cvpr.train-dense-frame-inventory.v1",
        "status": status,
        "scope": {
            "approved_video_count": expected_video_count,
            "verified_video_count": len(video_rows),
            "approved_image_rows": len(rows),
            "jpg_files": sum(row["jpg_files"] for row in video_rows),
            "frame_root": str(FRAME_ROOT),
        },
        "checks": {
            "all_approved_file_names_present": all(row["approved_file_names_present"] for row in video_rows),
            "all_frame_spans_count_match": all(row["frame_span_count_matches"] for row in video_rows),
            "all_approved_frame_offsets_match_listing_order": all(
                row["frame_order_matches_approved_offsets"] for row in video_rows
            ),
            "numeric_filename_sequences_without_gaps": all(
                row["numeric_file_names_contiguous"] is not False for row in video_rows
            ),
            "pixels_decoded_or_hashed": False,
        },
        "dataset_summary": dataset_summary,
        "video_rows": video_rows,
        "source": {
            "config": str(CONFIG.relative_to(ROOT)),
            "config_sha256": sha256_file(CONFIG),
            "image_plan": str(IMAGE_PLAN.relative_to(ROOT)),
            "image_plan_sha256": sha256_file(IMAGE_PLAN),
            "directory_listing_sha256": listing_digest.hexdigest(),
            "branch": git_value("branch", "--show-current"),
            "head": git_value("rev-parse", "HEAD"),
        },
        "boundary": config.get("semantic_access"),
        "violations": violations,
        "interpretation": (
            "Metadata-only dense-frame prerequisite: every approved Train video "
            "has the complete approved filename span and ordering, but this does "
            "not prove pixel decode, frontend output, semantic legality of a "
            "model, or any OCD/MOT result."
        ),
    }


def main() -> int:
    result = audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
