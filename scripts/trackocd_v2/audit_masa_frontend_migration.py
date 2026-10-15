#!/usr/bin/env python3
"""Audit the local historical MASA physical stream as a v2 migration candidate.

This is deliberately an eligibility audit, not a frontend conversion or a
metric runner.  It reads the sealed Val-only manifest, per-video completion
receipts, and the existing NPZ payloads one shard at a time.  It never opens
TAO annotations, GT/role files, images, Test assets, or model weights, and it
does not copy or rewrite the physical stream.  A report is printed by
default; ``--output`` is the only file this command may create.

The historical MASA stream is useful physical evidence, but it is not silently
promoted to the v2 bake-off: exact private training-stage binding, the v2
frontend stage receipt, and the v2 shared metric protocol must be proven
separately before a formal cache or selection can use it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "outputs/trackocd_core/physical/masa_full_val_annotated"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/v2/masa_frontend_migration_eligibility_v1.json"

MANIFEST_NAME = "prediction_manifest.json"
REQUIRED_NPZ_FIELDS = {
    "image_id",
    "frame_index",
    "frame_offsets",
    "track_id",
    "boxes",
    "score",
    "det_offsets",
    "det_boxes",
    "det_score",
}
FORBIDDEN_FIELD_TOKENS = (
    "category",
    "class",
    "vocab",
    "text",
    "gt",
    "role",
    "future",
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "json_root_is_not_object"
    return value, None


def _metadata(path: Path, *, hash_file: bool = False) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "is_symlink": path.is_symlink(),
    }
    if not path.is_file():
        return item
    try:
        stat = path.stat()
    except OSError as exc:
        item["stat_error"] = str(exc)
        return item
    item.update({"bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    if hash_file:
        item["sha256"] = _sha256(path)
    return item


def _finite_count(value: np.ndarray) -> int:
    if not np.issubdtype(value.dtype, np.number):
        return 0
    return int(np.size(value) - np.isfinite(value).sum())


def _offset_problem(offsets: np.ndarray, rows: int) -> str | None:
    if offsets.ndim != 1:
        return "offsets_not_1d"
    if offsets.size < 1:
        return "offsets_empty"
    if int(offsets[0]) != 0:
        return "offsets_do_not_start_at_zero"
    if int(offsets[-1]) != rows:
        return "offsets_last_value_mismatch"
    if np.any(np.diff(offsets) < 0):
        return "offsets_not_monotone"
    return None


def _audit_npz(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    """Read one small compressed shard and return aggregate-safe evidence."""

    result: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
        "required_fields": sorted(REQUIRED_NPZ_FIELDS),
        "fields": [],
        "status": "PASS",
        "problems": [],
        "finite_non_numeric_count": 0,
        "nonfinite_numeric_count": 0,
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = set(archive.files)
            result["fields"] = sorted(fields)
            missing = sorted(REQUIRED_NPZ_FIELDS - fields)
            extra = sorted(fields - REQUIRED_NPZ_FIELDS)
            if missing:
                result["problems"].append("missing_fields:" + ",".join(missing))
            if extra:
                result["extra_fields"] = extra
            forbidden = sorted(
                field
                for field in fields
                if any(token in field.lower() for token in FORBIDDEN_FIELD_TOKENS)
            )
            if forbidden:
                result["forbidden_semantic_like_fields"] = forbidden
                result["problems"].append("semantic_like_fields_present")
            if missing:
                result["status"] = "FAIL"
                return result

            arrays = {name: archive[name] for name in REQUIRED_NPZ_FIELDS}
            frame_count = int(arrays["image_id"].size)
            prediction_rows = int(arrays["track_id"].size)
            detection_rows = int(arrays["det_score"].size)
            result.update(
                {
                    "frames": frame_count,
                    "prediction_rows": prediction_rows,
                    "detection_rows": detection_rows,
                    "track_id_min": int(arrays["track_id"].min()) if prediction_rows else None,
                    "track_id_max": int(arrays["track_id"].max()) if prediction_rows else None,
                }
            )
            shape_checks = {
                "image_id": (frame_count,),
                "frame_index": (frame_count,),
                "frame_offsets": (frame_count + 1,),
                "track_id": (prediction_rows,),
                "boxes": (prediction_rows, 4),
                "score": (prediction_rows,),
                "det_offsets": (frame_count + 1,),
                "det_boxes": (detection_rows, 4),
                "det_score": (detection_rows,),
            }
            for name, expected in shape_checks.items():
                shape = tuple(int(value) for value in arrays[name].shape)
                if shape != expected:
                    result["problems"].append(f"shape_mismatch:{name}:{shape}!={expected}")

            frame_offset_problem = _offset_problem(arrays["frame_offsets"], prediction_rows)
            det_offset_problem = _offset_problem(arrays["det_offsets"], detection_rows)
            if frame_offset_problem:
                result["problems"].append("frame_offsets:" + frame_offset_problem)
            if det_offset_problem:
                result["problems"].append("det_offsets:" + det_offset_problem)

            for name in ("boxes", "score", "det_boxes", "det_score"):
                result["nonfinite_numeric_count"] += _finite_count(arrays[name])
            for name in ("image_id", "frame_index", "frame_offsets", "track_id", "det_offsets"):
                if not np.issubdtype(arrays[name].dtype, np.number):
                    result["finite_non_numeric_count"] += int(arrays[name].size)
            if result["nonfinite_numeric_count"]:
                result["problems"].append("nonfinite_numeric_values")
            if result["finite_non_numeric_count"]:
                result["problems"].append("non_numeric_index_values")
            for name in ("boxes", "score", "det_boxes", "det_score"):
                values = arrays[name]
                if np.issubdtype(values.dtype, np.number) and np.isfinite(values).all():
                    if name.endswith("boxes") and np.any(values[:, 2:] < values[:, :2]):
                        result["problems"].append("box_coordinates_degenerate_or_reversed:" + name)
            if prediction_rows != int(receipt.get("prediction_rows", -1)):
                result["problems"].append("prediction_rows_receipt_mismatch")
            if detection_rows != int(receipt.get("detection_rows", -1)):
                result["problems"].append("detection_rows_receipt_mismatch")
            if frame_count != int(receipt.get("images", -1)):
                result["problems"].append("image_count_receipt_mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["status"] = "FAIL"
        result["problems"].append("npz_read_error:" + str(exc))
    if result["problems"]:
        result["status"] = "FAIL"
    return result


def build_report(source_root: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    """Build a read-only migration report for a local MASA candidate."""

    source_root = Path(source_root)
    shard_root = source_root / "shards"
    manifest_path = source_root / MANIFEST_NAME
    manifest, manifest_error = _read_json(manifest_path) if manifest_path.is_file() else (None, "missing")
    physical_result_path = ROOT / "outputs/trackocd_core/audit/masa_full_val_physical_result.json"
    route_path = ROOT / "outputs/trackocd_core/audit/masa_training_route.json"
    physical_result, physical_result_error = _read_json(physical_result_path) if physical_result_path.is_file() else (None, "missing")
    route, route_error = _read_json(route_path) if route_path.is_file() else (None, "missing")

    problems: list[str] = []
    integrity_problems: list[str] = []
    if manifest is None:
        problems.append("prediction_manifest_missing_or_invalid")
    if physical_result is None:
        problems.append("physical_result_audit_missing_or_invalid")
    if route is None:
        problems.append("training_route_audit_missing_or_invalid")

    manifest_videos = manifest.get("videos", []) if isinstance(manifest, dict) else []
    manifest_by_id = {
        int(entry["video_id"]): entry
        for entry in manifest_videos
        if isinstance(entry, dict) and "video_id" in entry
    }
    receipts = sorted(shard_root.glob("*.complete.json")) if shard_root.is_dir() else []
    npz_files = sorted(shard_root.glob("*.npz")) if shard_root.is_dir() else []
    receipt_by_id: dict[int, dict[str, Any]] = {}
    receipt_errors: list[str] = []
    for path in receipts:
        receipt, error = _read_json(path)
        if receipt is None:
            receipt_errors.append(f"{path.name}:{error}")
            continue
        try:
            video_id = int(receipt["video_id"])
        except (KeyError, TypeError, ValueError):
            receipt_errors.append(f"{path.name}:video_id_missing_or_invalid")
            continue
        receipt_by_id[video_id] = receipt
    if receipt_errors:
        integrity_problems.extend(receipt_errors)

    shard_results: list[dict[str, Any]] = []
    npz_by_name = {path.name: path for path in npz_files}
    for video_id, entry in sorted(manifest_by_id.items()):
        receipt = receipt_by_id.get(video_id)
        if receipt is None:
            integrity_problems.append(f"video_{video_id}:completion_receipt_missing")
            continue
        if entry.get("status") != "COMPLETE_VIDEO" or receipt.get("status") != "COMPLETE_VIDEO":
            integrity_problems.append(f"video_{video_id}:not_complete")
        if entry.get("npz_filename") != receipt.get("npz_filename"):
            integrity_problems.append(f"video_{video_id}:manifest_receipt_filename_mismatch")
        npz_name = str(entry.get("npz_filename", ""))
        npz_path = npz_by_name.get(npz_name)
        if npz_path is None:
            integrity_problems.append(f"video_{video_id}:npz_missing")
            continue
        result = _audit_npz(npz_path, receipt)
        result["video_id"] = video_id
        if result.get("sha256") != receipt.get("npz_sha256"):
            result["problems"].append("npz_sha256_receipt_mismatch")
            result["status"] = "FAIL"
        if result.get("bytes") != int(receipt.get("npz_bytes", -1)):
            result["problems"].append("npz_bytes_receipt_mismatch")
            result["status"] = "FAIL"
        if result["status"] != "PASS":
            integrity_problems.extend(f"video_{video_id}:{problem}" for problem in result["problems"])
        shard_results.append(result)

    extra_receipt_ids = sorted(set(receipt_by_id) - set(manifest_by_id))
    if extra_receipt_ids:
        integrity_problems.append("completion_receipts_not_in_manifest:" + ",".join(map(str, extra_receipt_ids)))
    listed_npz = {str(entry.get("npz_filename")) for entry in manifest_by_id.values()}
    extra_npz = sorted(path.name for path in npz_files if path.name not in listed_npz)
    if extra_npz:
        integrity_problems.append("npz_files_not_in_manifest:" + ",".join(extra_npz))

    manifest_flags = {
        "status": manifest.get("status") if manifest else None,
        "schema_version": manifest.get("schema_version") if manifest else None,
        "test_access": manifest.get("test_access") if manifest else None,
        "gt_or_role_model_input": manifest.get("gt_or_role_model_input") if manifest else None,
        "training": manifest.get("training") if manifest else None,
        "primary_freeze_permitted": manifest.get("primary_freeze_permitted") if manifest else None,
    }
    for key, expected in (
        ("status", "SEALED_COMPLETE_FULL_VAL_PHYSICAL_STREAM"),
        ("test_access", False),
        ("gt_or_role_model_input", False),
        ("training", False),
        ("primary_freeze_permitted", False),
    ):
        if manifest_flags.get(key) != expected:
            problems.append(f"manifest_{key}_not_{expected!r}")

    provenance = (physical_result or {}).get("provenance", {})
    route_flags = {
        "physical_result_status": (physical_result or {}).get("status"),
        "physical_result_primary_freeze_permitted": (physical_result or {}).get("primary_freeze_permitted"),
        "exact_private_sam_release_training_stage_binding_verified": provenance.get(
            "exact_private_sam_release_training_stage_binding_verified"
        ),
        "forbidden_supervision_proven": provenance.get("forbidden_supervision_proven"),
        "training_route_status": (route or {}).get("status"),
        "training_route_primary_freeze_permitted": (route or {}).get("primary_freeze_permitted_by_this_audit"),
        "training_route_exact_stage_binding": (route or {}).get("exact_checkpoint_stage_config_supervision_binding_verified"),
    }
    # These are explicit non-admission facts, not inferred failures of the
    # runtime boundary.  The stream is preserved as a physical candidate.
    if route_flags["physical_result_primary_freeze_permitted"] is not False:
        problems.append("historical_physical_result_did_not_explicitly_deny_primary_freeze")
    if route_flags["exact_private_sam_release_training_stage_binding_verified"] is not False:
        problems.append("exact_private_training_stage_binding_not_explicitly_unverified")
    if route_flags["training_route_exact_stage_binding"] is not False:
        problems.append("exact_checkpoint_stage_config_supervision_binding_not_explicitly_unverified")

    passed_shards = sum(result.get("status") == "PASS" for result in shard_results)
    total_prediction_rows = sum(int(result.get("prediction_rows", 0)) for result in shard_results)
    total_detection_rows = sum(int(result.get("detection_rows", 0)) for result in shard_results)
    total_images = sum(int(result.get("frames", 0)) for result in shard_results)
    source_metadata = {
        "root": str(source_root),
        "resolved_root": str(source_root.resolve(strict=False)),
        "manifest": _metadata(manifest_path, hash_file=True),
        "shard_count": len(npz_files),
        "completion_receipt_count": len(receipts),
        "npz_total_bytes": sum(int(path.stat().st_size) for path in npz_files if path.is_file()),
    }
    stream_contract = {
        "historical_scope": "M1 full-Val frozen physical candidate audit on annotated cadence",
        "dense_frame_online_proven": False,
        "same_v2_predicted_stream_proven": False,
        "same_v2_evaluator_contract_proven": False,
        "native_row_fields_representable": sorted(
            {"video_id", "image_id", "frame_index", "physical_track_id", "bbox_xyxy", "score"}
        ),
        "category_free_payload_fields_observed": sorted(REQUIRED_NPZ_FIELDS),
        "appearance_embedding_payload_present": False,
        "conversion_or_copy_performed": False,
    }
    admission_reasons = [
        "historical stream is complete and integrity-checked, but its receipt scope is annotated-cadence rather than a proven dense frame-online v2 stream",
        "primary_freeze_permitted is false in the sealed physical audit",
        "exact private SAM release/training-stage binding is not verified",
        "no v2 frontend stage receipt or shared v2 Val metric contract is attached",
    ]
    if integrity_problems:
        admission_reasons.insert(0, "one or more physical shard/receipt integrity checks failed")
    report = {
        "schema_version": "trackocd.v2.masa_frontend_migration_eligibility.v1",
        "status": "HISTORICAL_MASA_PHYSICAL_CANDIDATE_NOT_V2_ADMISSIBLE",
        "generated_utc": _utc_now(),
        "read_only_audit": True,
        "admissible_for_formal_v2": False,
        "primary_frontend_qualification": False,
        "formal_common_feature_cache_authorized": False,
        "candidate_role": "HISTORICAL_PHYSICAL_FRONTEND_CANDIDATE_ONLY",
        "source": source_metadata,
        "manifest_flags": manifest_flags,
        "route_and_provenance": route_flags,
        "stream_contract": stream_contract,
        "coverage": {
            "manifest_videos": len(manifest_videos),
            "manifest_images": manifest.get("images") if manifest else None,
            "manifest_detection_rows": manifest.get("total_detection_rows") if manifest else None,
            "manifest_prediction_rows": manifest.get("total_prediction_rows") if manifest else None,
            "shards_passed": passed_shards,
            "shards_total": len(shard_results),
            "audited_images": total_images,
            "audited_detection_rows": total_detection_rows,
            "audited_prediction_rows": total_prediction_rows,
        },
        "integrity": {
            "status": "PASS" if not integrity_problems else "FAIL",
            "problems": integrity_problems,
            "shard_results_omitted": True,
            "shard_result_digest": hashlib.sha256(
                json.dumps(
                    [
                        {
                            "video_id": result.get("video_id"),
                            "status": result.get("status"),
                            "sha256": result.get("sha256"),
                            "bytes": result.get("bytes"),
                            "problems": result.get("problems", []),
                        }
                        for result in shard_results
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        "admission_reasons": admission_reasons,
        "source_boundary": {
            "tao_test_accessed": False,
            "gt_or_role_model_input": False,
            "training_performed": False,
            "model_weights_loaded": False,
            "images_decoded": False,
            "network_used": False,
            "external_processes_touched": False,
            "source_stream_modified": False,
            "historical_metrics_imported_for_admission": False,
        },
        "errors": problems,
        "next_allowed_action": "Keep the MASA stream quarantined as physical evidence; only after an explicit v2 route receipt, exact provenance decision, and shared Val metric contract may a separate normalized candidate be registered.",
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = build_report(args.source_root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, args.output)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
