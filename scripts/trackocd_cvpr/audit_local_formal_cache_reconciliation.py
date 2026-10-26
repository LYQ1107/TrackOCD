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
LOCAL_CURRENT_PARTIAL_AUDIT = "outputs/trackocd_v2_runtime/audit/local_399_partial_cache_payload_audit_20261026.json"
HISTORICAL_AUDIT = "outputs/trackocd_v2_runtime/audit/formal_cache_payload_audit_20261022.json"
RANGE_MANIFEST_GLOB = "cache_manifest.worker-*.json"


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


def _range_manifest_candidates(manifest_path: Path, manifest: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    """Load only small cumulative range manifests bound to this cache.

    The canonical manifest is intentionally regenerated only after all shards
    finish.  During a resumable continuation, worker manifests are the
    authoritative metadata for newly written atomic markers; they must not be
    confused with a formal completion proof.
    """

    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(manifest_path.parent.glob(RANGE_MANIFEST_GLOB)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("cache_root") != manifest.get("cache_root"):
            continue
        if payload.get("frontend") != manifest.get("frontend") or payload.get("split") != manifest.get("split"):
            continue
        if not isinstance(payload.get("completed_shards"), list):
            continue
        candidates.append((path, payload))
    return candidates


def _marker_metadata(cache_root: Path, shard_tracks: int, source_sha256: str) -> tuple[list[int], list[str]]:
    """Read shard completion markers only; never open payload arrays."""

    indices: list[int] = []
    errors: list[str] = []
    shard_root = cache_root / "shards"
    if not shard_root.is_dir():
        return indices, errors
    for marker in sorted(shard_root.glob("shard-*/.done")):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"invalid_marker:{marker}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"non_object_marker:{marker}")
            continue
        try:
            track_start = int(payload["track_start"])
            track_count = int(payload["track_count"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"missing_marker_fields:{marker}")
            continue
        if track_count != shard_tracks or track_start < 0 or track_start % shard_tracks:
            errors.append(f"invalid_marker_range:{marker}")
            continue
        if payload.get("source_sha256") != source_sha256:
            errors.append(f"marker_source_mismatch:{marker}")
            continue
        indices.append(track_start // shard_tracks)
    return sorted(set(indices)), errors


def _select_partial_audit(
    repo: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    range_candidates: list[tuple[Path, dict[str, Any]]],
) -> tuple[Path, dict[str, Any]]:
    """Select the most complete current partial payload audit.

    The first continuation audit covered only the canonical 384-entry
    manifest.  A later bounded audit may cover the cumulative range manifest
    and all currently marked shards.  Prefer that newer, strictly bound
    receipt, while retaining the legacy fallback for recovery of older local
    snapshots.
    """

    local_root = str(Path(str(manifest.get("cache_root", ""))).resolve())
    source_sha256 = str((manifest.get("input") or {}).get("sha256") or "")
    allowed_manifests = {manifest_path.resolve()}
    allowed_manifests.update(path.resolve() for path, _ in range_candidates)
    candidates = (
        LOCAL_CURRENT_PARTIAL_AUDIT,
        LOCAL_PARTIAL_AUDIT,
    )
    valid: list[tuple[int, str, Path, dict[str, Any]]] = []
    for relative_path in candidates:
        path = repo / relative_path
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            bound_manifest = Path(str(payload.get("manifest", ""))).resolve()
            completed = int(payload.get("completed_shards", -1))
        except (TypeError, ValueError, OSError):
            continue
        if (
            bound_manifest not in allowed_manifests
            or str(payload.get("cache_root", "")) != local_root
            or payload.get("input_stream_sha256") != source_sha256
            or payload.get("frontend") != manifest.get("frontend")
            or payload.get("split") != manifest.get("split")
            or payload.get("status") != "PASS_PARTIAL_COMPLETED_SHARDS"
            or payload.get("error_count") != 0
            or payload.get("invalid_box_count") != 0
            or payload.get("test_content_opened") is not False
            or payload.get("external_processes_touched") is not False
        ):
            continue
        valid.append((completed, str(payload.get("generated_utc", "")), path, payload))
    if valid:
        _, _, path, payload = max(valid, key=lambda item: (item[0], item[1], str(item[2])))
        return path, payload
    return load(repo, LOCAL_PARTIAL_AUDIT)


def build_audit(repo: Path) -> dict[str, Any]:
    manifest_path, manifest = load(repo, LOCAL_MANIFEST)
    historical_path, historical = load(repo, HISTORICAL_AUDIT)

    shard_tracks = int(manifest["shard_tracks"])
    total_tracks = int(manifest["total_tracks"])
    required_shards = (total_tracks + shard_tracks - 1) // shard_tracks
    base_entries = manifest.get("completed_shards") or []
    base_completed_indices = sorted(int(entry["track_start"]) // shard_tracks for entry in base_entries)

    range_candidates = _range_manifest_candidates(manifest_path, manifest)
    partial_path, partial = _select_partial_audit(repo, manifest_path, manifest, range_candidates)
    latest_range_path: Path | None = None
    latest_range: dict[str, Any] | None = None
    if range_candidates:
        latest_range_path, latest_range = max(
            range_candidates,
            key=lambda item: (
                len(item[1].get("completed_shards") or []),
                str(item[1].get("generated_utc", "")),
                item[0].name,
            ),
        )
    observed_entries = (latest_range or {}).get("completed_shards") or base_entries
    observed_completed_indices = sorted(int(entry["track_start"]) // shard_tracks for entry in observed_entries)
    observed_manifest_count = len(observed_entries)
    marker_indices, marker_errors = _marker_metadata(
        Path(str(manifest.get("cache_root"))).resolve(), shard_tracks, str(manifest["input"]["sha256"])
    )
    completed_indices = observed_completed_indices
    completed_set = set(completed_indices)
    missing_indices = [index for index in range(required_shards) if index not in completed_set]

    historical_manifest = Path(str(historical.get("manifest", "")))
    historical_root = Path(str(historical.get("cache_root", "")))
    local_manifest_resolved = str(manifest_path.resolve())
    allowed_manifest_paths = {manifest_path.resolve()}
    allowed_manifest_paths.update(path.resolve() for path, _ in range_candidates)
    partial_manifest_matches = Path(str(partial.get("manifest", ""))).resolve() in allowed_manifest_paths
    local_root = Path(str(manifest.get("cache_root", ""))).resolve()
    local_root_matches = str(partial.get("cache_root", "")) == str(local_root)
    partial_checked_track_count = int(partial.get("checked_tracks", -1))
    base_checked_track_count = sum(
        int(entry["track_count"]) for entry in base_entries
    )
    observed_checked_track_count = sum(int(entry["track_count"]) for entry in observed_entries)
    partial_completed_count_matches_base = len(base_entries) == int(partial.get("completed_shards", -1))
    partial_checked_track_count_matches_base = partial_checked_track_count == base_checked_track_count
    partial_completed_count_matches_observed = observed_manifest_count == int(partial.get("completed_shards", -1))
    partial_checked_track_count_matches_observed = partial_checked_track_count == observed_checked_track_count
    observed_observation_count = sum(int(entry.get("observation_count", 0)) for entry in observed_entries)
    marker_set = set(marker_indices)
    observed_set = set(observed_completed_indices)
    range_manifest_matches_markers = latest_range is not None and observed_set == marker_set

    local_complete = (
        manifest.get("status") == "COMPLETE"
        and observed_manifest_count == required_shards
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
        and (
            (partial_completed_count_matches_base and partial_checked_track_count_matches_base)
            or (partial_completed_count_matches_observed and partial_checked_track_count_matches_observed)
        )
        and range_manifest_matches_markers
        and not marker_errors
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
            "manifest_completed_shards": len(base_completed_indices),
            "observed_range_manifest": str(latest_range_path.relative_to(repo)) if latest_range_path else None,
            "observed_range_manifest_status": latest_range.get("status") if latest_range else None,
            "observed_completed_shards": observed_manifest_count,
            "observed_completed_shard_ranges": ranges(observed_completed_indices),
            "observed_observations": observed_observation_count,
            "marker_completed_shards": len(marker_indices),
            "marker_completed_shard_ranges": ranges(marker_indices),
            "marker_errors": marker_errors,
            "frontend": manifest.get("frontend"),
            "split": manifest.get("split"),
            "shard_tracks": shard_tracks,
            "total_tracks": total_tracks,
            "required_shards": required_shards,
            "completed_shards": len(completed_indices),
            "completed_shard_ranges": ranges(completed_indices),
            "missing_shards": len(missing_indices),
            "missing_shard_ranges": ranges(missing_indices),
            "checked_tracks": partial_checked_track_count,
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
            "base_manifest_count_matches_partial_audit": partial_completed_count_matches_base,
            "base_checked_track_count_matches_manifest_entries": partial_checked_track_count_matches_base,
            "observed_range_count_matches_partial_audit": partial_completed_count_matches_observed,
            "observed_range_checked_track_count_matches_partial_audit": partial_checked_track_count_matches_observed,
            "range_manifest_matches_completion_markers": range_manifest_matches_markers,
            "completion_marker_errors": marker_errors,
            "partial_payload_audit_covers_observed_range": partial_completed_count_matches_observed
            and partial_checked_track_count_matches_observed,
            "partial_payload_audit_covers_base_range": partial_completed_count_matches_base
            and partial_checked_track_count_matches_base,
            "local_formal_cache_complete": local_complete,
            "historical_cache_reachable": historical_reachable,
        },
        "evidence": [
            evidence(manifest_path, LOCAL_MANIFEST, manifest),
            evidence(partial_path, str(partial_path.relative_to(repo)), partial),
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
