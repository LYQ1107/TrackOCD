#!/usr/bin/env python3
"""Stream-audit the four fixed anonymous COVTrack Train dense shards.

The audit consumes only the committed input audits, compact run receipts, and
the normalized category-free ``physical_tracks.jsonl`` streams.  It never
opens native evaluator rows, TAO GT arrays, or the public vocabulary.  The
physical streams are read line by line so the audit does not materialize the
full prediction cache in memory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/trackocd_cvpr/joint_causal_pred_dense_protocol_v1.json"
DEFAULT_IMAGE_PLAN = ROOT / "outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v12_images/private_exact_joint_proof_or_assignment.json"
DEFAULT_SHARD_ROOT = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_shards_v1"
DEFAULT_OUTPUT = DEFAULT_SHARD_ROOT / "aggregate_audit.json"
COMPLETE_RECEIPT_STATUSES = {
    "COMPLETE_ANONYMOUS_TRAIN_DENSE_FRONTEND_SHARD",
    "COMPLETE_ANONYMOUS_TRAIN_DENSE_FRONTEND_SHARD_TCC_OFF",
}
PROTOCOL_STATUSES = {
    "FROZEN_CAUSAL_BOUNDARIES_PRED_CACHE_PENDING",
    "FROZEN_CAUSAL_BOUNDARIES_FULL_PRED_CACHE_COMPLETE_FRONTEND_QUALIFICATION_PENDING",
}

PHYSICAL_FIELDS = {
    "sample_key",
    "source_sample_id",
    "video_id",
    "physical_track_id",
    "frame_ids",
    "boxes_xyxy",
    "image_paths",
    "quality",
    "stream_order",
    "source_split",
}
FORBIDDEN_FIELD_PARTS = ("category", "semantic", "text", "gt", "label", "novel", "prototype")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_with_sha(path: Path) -> Iterator[tuple[dict[str, Any], int]]:
    """Yield JSONL rows and the byte offset at which each row started."""

    with path.open("rb") as handle:
        offset = 0
        for line_number, raw in enumerate(handle, 1):
            start = offset
            offset += len(raw)
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            yield value, start


def _plan_video_ids(image_plan: Path) -> list[int]:
    payload = _json(image_plan)
    return sorted({int(row["video_id"]) for row in payload.get("image_metadata", [])})


def _finite_box(box: Any) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in box)
        and float(box[2]) > float(box[0])
        and float(box[3]) > float(box[1])
    )


def _audit_physical_stream(path: Path, expected_video_ids: set[int]) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path.resolve())}

    observations = 0
    invalid_boxes = 0
    tracks = 0
    video_ids: set[int] = set()
    track_keys: set[tuple[int, str]] = set()
    invalid_examples: list[str] = []
    errors: list[str] = []

    for row, offset in _jsonl_with_sha(path):
        fields = set(row)
        if fields != PHYSICAL_FIELDS:
            errors.append(f"row at byte {offset} has fields {sorted(fields)}")
            if len(errors) >= 5:
                break
        for field in fields:
            lowered = field.lower()
            if any(part in lowered for part in FORBIDDEN_FIELD_PARTS):
                errors.append(f"forbidden physical field {field!r}")
                break
        try:
            video_id = int(row["video_id"])
            track_id = str(row["physical_track_id"])
            frame_ids = row["frame_ids"]
            boxes = row["boxes_xyxy"]
            if row["source_split"] != "train_predicted":
                errors.append(f"source_split is not train_predicted at byte {offset}")
            if video_id not in expected_video_ids:
                errors.append(f"video {video_id} is outside the registered shard at byte {offset}")
            if not isinstance(frame_ids, list) or not isinstance(boxes, list) or len(frame_ids) != len(boxes):
                errors.append(f"frame/box length mismatch at byte {offset}")
                continue
            if len(set(frame_ids)) != len(frame_ids):
                errors.append(f"duplicate frame in physical track at byte {offset}")
            key = (video_id, track_id)
            if key in track_keys:
                errors.append(f"duplicate physical track row for {key}")
            track_keys.add(key)
            tracks += 1
            video_ids.add(video_id)
            for index, box in enumerate(boxes):
                observations += 1
                if not _finite_box(box):
                    invalid_boxes += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append(f"{video_id}/{track_id}/{index}")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"malformed row at byte {offset}: {type(exc).__name__}")
        if len(errors) >= 5:
            break

    result = {
        "status": "PASS" if not errors else "FAIL",
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "tracks": tracks,
        "observations": observations,
        "videos_with_rows": len(video_ids),
        "video_ids_with_rows": sorted(video_ids),
        "invalid_boxes": invalid_boxes,
        "invalid_examples": invalid_examples,
        "errors": errors[:5],
    }
    return result


def audit(
    config_path: Path,
    image_plan_path: Path,
    shard_root: Path,
    output_path: Path,
    receipt_roots: list[Path] | None = None,
) -> dict[str, Any]:
    config = _json(config_path)
    plan_ids = _plan_video_ids(image_plan_path)
    config_shards = config.get("predicted_dense_cache", {}).get("shards", [])
    errors: list[str] = []
    shard_results: list[dict[str, Any]] = []
    all_input_ids: list[int] = []
    seen_input_ids: set[int] = set()
    complete = True

    if config.get("status") not in PROTOCOL_STATUSES:
        errors.append("unexpected protocol status")
    if len(plan_ids) != 407 or len(config_shards) != 4:
        errors.append(f"expected 407 approved IDs and 4 config shards, got {len(plan_ids)} and {len(config_shards)}")

    for index, shard in enumerate(config_shards):
        shard_dir = shard_root / f"shard_{index}"
        input_path = shard_dir / "input_audit.json"
        receipt_root = receipt_roots[index] if receipt_roots is not None else shard_dir
        run_path = receipt_root / "run.json"
        shard_record: dict[str, Any] = {"shard": index, "input_audit": str(input_path.resolve())}
        shard_record["receipt_root"] = str(receipt_root.resolve())
        if not input_path.is_file():
            errors.append(f"missing input audit for shard {index}")
            complete = False
            shard_results.append(shard_record)
            continue

        input_audit = _json(input_path)
        selection = input_audit.get("selection", {})
        ids = [int(value) for value in selection.get("video_ids", [])]
        all_input_ids.extend(ids)
        duplicates = sorted(set(ids) & seen_input_ids)
        if duplicates:
            errors.append(f"input video overlap in shard {index}: {duplicates[:5]}")
        seen_input_ids.update(ids)
        if input_audit.get("status") != "PASS_FIXED_ANONYMOUS_TRAIN_DENSE_INPUT":
            errors.append(f"input audit status failed for shard {index}")
        if int(selection.get("shard_index", -1)) != index:
            errors.append(f"input audit shard index mismatch for shard {index}")
        if ids != plan_ids[int(shard["ordinal_min"]) : int(shard["ordinal_max"]) + 1]:
            errors.append(f"input video-ID order mismatch for shard {index}")
        for boundary in ("model_executed", "training_started", "gt_or_semantic_arrays_read", "tao_val_accessed", "tao_test_accessed", "pixels_decoded_or_hashed"):
            if input_audit.get("boundaries", {}).get(boundary) is not False:
                errors.append(f"input boundary {boundary} is not false for shard {index}")

        shard_record["input_video_count"] = len(ids)
        shard_record["input_video_id_first"] = ids[0] if ids else None
        shard_record["input_video_id_last"] = ids[-1] if ids else None
        shard_record["input_audit_sha256"] = _sha256(input_path)
        if not run_path.is_file():
            shard_record["status"] = "RUNNING_OR_INCOMPLETE"
            complete = False
            shard_results.append(shard_record)
            continue

        receipt = _json(run_path)
        shard_record["run_receipt"] = str(run_path.resolve())
        shard_record["run_receipt_sha256"] = _sha256(run_path)
        shard_record["status"] = receipt.get("status")
        if receipt.get("status") not in COMPLETE_RECEIPT_STATUSES:
            complete = False
            shard_results.append(shard_record)
            continue
        for key, expected in (("split", "train_predicted"), ("training_started", False), ("formal_metrics", False), ("tao_val_accessed", False), ("tao_test_accessed", False), ("gt_or_semantic_arrays_read", False)):
            if receipt.get(key) != expected:
                errors.append(f"receipt {key} boundary failed for shard {index}")
        physical = Path(receipt.get("physical_stream", ""))
        stream_result = _audit_physical_stream(physical, set(ids))
        shard_record["physical_stream"] = stream_result
        counts = receipt.get("counts", {})
        for key in ("tracks", "observations", "invalid_boxes"):
            if key in counts and key in stream_result and int(counts[key]) != int(stream_result[key]):
                errors.append(f"receipt/stream {key} mismatch for shard {index}")
        if stream_result.get("status") != "PASS":
            errors.append(f"physical stream audit failed for shard {index}")
        shard_results.append(shard_record)

    if sorted(all_input_ids) != plan_ids:
        errors.append("union of fixed input video IDs does not equal approved 407-video order")
    status = "PASS_COMPLETE_FOUR_FIXED_ANONYMOUS_TRAIN_DENSE_SHARDS" if complete and not errors else ("WAITING_FOR_FOUR_SHARD_RECEIPTS" if complete is False and not errors else "FAIL_DENSE_SHARD_AUDIT")
    result = {
        "schema_version": "trackocd.cvpr.covtrack-train-dense-shard-audit.v1",
        "status": status,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": str(config_path.resolve()),
        "protocol_sha256": _sha256(config_path),
        "approved_image_plan": str(image_plan_path.resolve()),
        "approved_image_plan_sha256": _sha256(image_plan_path),
        "approved_video_count": len(plan_ids),
        "input_video_count": len(all_input_ids),
        "input_video_order_sha256": hashlib.sha256(json.dumps(sorted(all_input_ids), separators=(",", ":")).encode()).hexdigest() if all_input_ids else None,
        "shards": shard_results,
        "errors": errors[:20],
        "boundaries": {
            "training_started": False,
            "formal_metrics_created": False,
            "tao_val_or_test_labels_used": False,
            "native_evaluator_rows_read": False,
            "private_gt_arrays_read": False,
        },
        "interpretation": "Dense predicted frontend coverage/provenance audit only; no OCD/MOT metric or Goal A/B/C acceptance is claimed.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image-plan", type=Path, default=DEFAULT_IMAGE_PLAN)
    parser.add_argument("--shard-root", type=Path, default=DEFAULT_SHARD_ROOT)
    parser.add_argument(
        "--receipt-root-template",
        help=(
            "optional per-shard receipt directory template; use {index} for 0..3. "
            "Without it, receipts are read from --shard-root/shard_{index}."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    receipt_roots = None
    if args.receipt_root_template:
        receipt_roots = [
            Path(args.receipt_root_template.format(index=index)).resolve()
            for index in range(4)
        ]
    result = audit(
        args.config.resolve(),
        args.image_plan.resolve(),
        args.shard_root.resolve(),
        args.output.resolve(),
        receipt_roots=receipt_roots,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_complete and result["status"] != "PASS_COMPLETE_FOUR_FIXED_ANONYMOUS_TRAIN_DENSE_SHARDS":
        return 2
    return 0 if result["status"] != "FAIL_DENSE_SHARD_AUDIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
