#!/usr/bin/env python3
"""Reconcile the reachable local Val cache with the historical NAS receipt.

Only small manifests and prior compact audit receipts are opened.  Feature
arrays, streams, images, annotations, labels, weights, and processes are not
opened or modified.  A historical PASS receipt is never promoted when its
recorded NAS path is currently unreachable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "trackocd.cvpr.local-formal-cache-reconciliation.v1"
LOCAL_MANIFEST = "outputs/trackocd_v2_runtime/features/formal/covtrack_native_continuation_20261022/cache_manifest.json"
LOCAL_PARTIAL_AUDIT = "outputs/trackocd_v2_runtime/audit/local_partial_cache_payload_audit_20261026.json"
HISTORICAL_AUDIT = "outputs/trackocd_v2_runtime/audit/formal_cache_payload_audit_20261022.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(repo: Path, relative_path: str) -> tuple[Path, dict[str, Any]]:
    path = repo / relative_path
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {relative_path}")
    return path, value


def evidence(path: Path, relative_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": relative_path,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
        "reported_status": payload.get("status"),
    }


def ranges(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    output: list[list[int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            output.append([start, previous])
            start = value
        previous = value
    output.append([start, previous])
    return output


def build_audit(repo: Path) -> dict[str, Any]:
    manifest_path, manifest = load(repo, LOCAL_MANIFEST)
    partial_path, partial = load(repo, LOCAL_PARTIAL_AUDIT)
    historical_path, historical = load(repo, HISTORICAL_AUDIT)

    shard_tracks = int(manifest["shard_tracks"])
    total_tracks = int(manifest["total_tracks"])
    required_shards = (total_tracks + shard_tracks - 1) // shard_tracks
    entries = manifest.get("completed_shards") or []
    completed_indices = sorted(int(entry["track_start"]) // shard_tracks for entry in entries)
    completed_set = set(completed_indices)
    missing_indices = [index for index in range(required_shards) if index not in completed_set]

    historical_manifest = Path(str(historical.get("manifest", "")))
    historical_root = Path(str(historical.get("cache_root", "")))
    local_manifest_resolved = str(manifest_path.resolve())
    partial_manifest_matches = str(partial.get("manifest", "")) == local_manifest_resolved
    local_root = Path(str(manifest.get("cache_root", ""))).resolve()
    local_root_matches = str(partial.get("cache_root", "")) == str(local_root)
    manifest_count_matches = len(entries) == int(partial.get("completed_shards", -1))
    complete_count_matches = int(partial.get("checked_tracks", -1)) == sum(
        int(entry["track_count"]) for entry in entries
    )

    local_complete = (
        manifest.get("status") == "COMPLETE"
        and len(entries) == required_shards
        and partial.get("formal_cache_complete") is True
    )
    historical_reachable = historical_manifest.is_file() and historical_root.is_dir()
    status = (
        "PASS_LOCAL_PARTIAL_CACHE_RECONCILED_HISTORICAL_COMPLETE_CACHE_UNREACHABLE_NOT_FORMAL_READY"
        if manifest.get("frontend") == "COVTrack-native"
        and manifest.get("split") == "val"
        and manifest.get("test_semantic_accessed") is False
        and partial.get("status") == "PASS_PARTIAL_COMPLETED_SHARDS"
        and partial.get("error_count") == 0
        and partial.get("invalid_box_count") == 0
        and partial.get("test_content_opened") is False
        and partial.get("external_processes_touched") is False
        and partial_manifest_matches
        and local_root_matches
        and manifest_count_matches
        and complete_count_matches
        and not local_complete
        and not historical_reachable
        and len(missing_indices) > 0
        else "INCOMPLETE_LOCAL_FORMAL_CACHE_RECONCILIATION"
    )

    return {
        "schema_version": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": {
            "small_metadata_receipts_only": True,
            "feature_arrays_opened": False,
            "large_streams_opened": False,
            "data_or_weights_read": False,
            "tao_test_read": False,
            "external_processes_touched": False,
            "model_selection_performed": False,
        },
        "local_cache": {
            "manifest": LOCAL_MANIFEST,
            "manifest_status": manifest.get("status"),
            "frontend": manifest.get("frontend"),
            "split": manifest.get("split"),
            "shard_tracks": shard_tracks,
            "total_tracks": total_tracks,
            "required_shards": required_shards,
            "completed_shards": len(completed_indices),
            "completed_shard_ranges": ranges(completed_indices),
            "missing_shards": len(missing_indices),
            "missing_shard_ranges": ranges(missing_indices),
            "checked_tracks": partial.get("checked_tracks"),
            "checked_observations": partial.get("checked_observations"),
            "invalid_box_count": partial.get("invalid_box_count"),
            "formal_cache_complete": local_complete,
            "cache_root_reachable": local_root.is_dir(),
        },
        "historical_cache": {
            "audit": HISTORICAL_AUDIT,
            "reported_status": historical.get("status"),
            "reported_required_shards": historical.get("required_shards"),
            "reported_completed_shards": historical.get("completed_shards"),
            "manifest_path": str(historical_manifest),
            "cache_root": str(historical_root),
            "manifest_reachable_now": historical_manifest.is_file(),
            "cache_root_reachable_now": historical_root.is_dir(),
            "historical_pass_is_current_asset": False,
        },
        "binding_checks": {
            "partial_manifest_matches_local_manifest": partial_manifest_matches,
            "partial_audit_cache_root_matches_manifest": local_root_matches,
            "manifest_count_matches_partial_audit": manifest_count_matches,
            "checked_track_count_matches_manifest_entries": complete_count_matches,
            "local_formal_cache_complete": local_complete,
            "historical_cache_reachable": historical_reachable,
        },
        "evidence": [
            evidence(manifest_path, LOCAL_MANIFEST, manifest),
            evidence(partial_path, LOCAL_PARTIAL_AUDIT, partial),
            evidence(historical_path, HISTORICAL_AUDIT, historical),
        ],
        "scientific_boundary": "This reconciles the current cache checkpoint only; it does not select a frontend, authorize formal predicted evaluation, or claim OCD/MOT or Goal A/B/C success.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trackocd_cvpr/cvpr5/local_formal_cache_reconciliation_20261026.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    result = build_audit(repo)
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0 if result["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
