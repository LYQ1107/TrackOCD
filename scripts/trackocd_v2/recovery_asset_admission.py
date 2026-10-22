#!/usr/bin/env python3
"""Read-only admission audit for recovered TrackOCD v2 assets.

This checker answers a deliberately narrow question: do the exact v2 inputs
needed by the frozen predicted evaluator exist and have consistent small
metadata?  It is safe to run during migration because it does not create an
output layout, open images or annotations, read the public JSONL stream, load
feature arrays, access TAO Test, use the network, or inspect/modify processes.

Large inputs are represented by ``stat`` evidence only.  Small JSON control
files and ``.done`` markers are parsed and hashed.  A JSON report is printed
to stdout; ``--output`` is an explicit opt-in for writing one report file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_SMALL_JSON_BYTES = 16 * 1024 * 1024
FRONTEND_SLUGS = {
    "SimOWT/Q0": "simowt",
    "OVTR-native": "ovtr",
    "COVTrack-native": "covtrack_native",
    "COVTrack-NoSemantic": "covtrack_nosem",
}
REQUIRED_FRONTEND = "COVTrack-native"
ROLE_FILES = (
    ("known_ids", "splits/known_ids.json"),
    ("unknown_ids_val", "splits/unknown_ids_val.json"),
    ("distractor_ids", "splits/distractor_ids.json"),
)
CAUSAL_ORDER_FILES = (
    "val_gt_track_stream.jsonl",
    "val_gt_track_stream_seed1027.jsonl",
    "val_gt_track_stream_seed1028.jsonl",
    "val_gt_track_stream_seed1029.jsonl",
)
SUPPLEMENTAL_FRONTEND_FILES = (
    ("covtrack_native_native_run", "audit/frontend_covtrack_native_native_run.json"),
    ("covtrack_native_metrics", "audit/frontend_covtrack_native_metrics.json"),
)
REQUIRED_STAGE_FIELDS = (
    "native_run_complete",
    "physical_stream_contract_complete",
    "same_v2_physical_metric_protocol",
    "same_v2_evaluator_contract",
    "requested_metrics_complete",
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved(path: Path) -> str:
    return str(path.resolve(strict=False))


def _asset(path: Path, role: str, *, read_json: bool = False, hash_file: bool = False) -> dict[str, Any]:
    """Collect bounded metadata without creating or following arbitrary trees."""

    item: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "exists": path.is_file(),
        "lexists": path.exists() or path.is_symlink(),
        "is_symlink": path.is_symlink(),
        "resolved_path": _resolved(path),
        "opened": False,
    }
    if not path.is_file():
        return item
    try:
        stat = path.stat()
        item["size_bytes"] = int(stat.st_size)
        item["mtime_ns"] = int(stat.st_mtime_ns)
    except OSError as exc:
        item["stat_error"] = str(exc)
        return item
    if hash_file:
        if item["size_bytes"] > MAX_SMALL_JSON_BYTES:
            item["hash_skipped"] = "file_exceeds_small_metadata_limit"
        else:
            item["sha256"] = _sha256(path)
            item["opened"] = True
    if read_json:
        if item["size_bytes"] > MAX_SMALL_JSON_BYTES:
            item["read_error"] = "json_exceeds_small_metadata_limit"
        else:
            try:
                item["payload"] = json.loads(path.read_text(encoding="utf-8"))
                item["opened"] = True
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                item["read_error"] = str(exc)
    return item


def _safe_payload(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload")
    return payload if isinstance(payload, dict) else {}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _git_info(project_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"project_root": str(project_root), "read_only": True}
    for key, args in (
        ("branch", ("rev-parse", "--abbrev-ref", "HEAD")),
        ("head", ("rev-parse", "HEAD")),
        ("status", ("status", "--porcelain")),
    ):
        try:
            proc = subprocess.run(
                ["git", "-C", str(project_root), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[key + "_error"] = str(exc)
            continue
        if proc.returncode == 0:
            value = proc.stdout.strip()
            result[key] = value
        else:
            result[key + "_error"] = (proc.stderr or proc.stdout).strip()[-1000:]
    return result


def _data_assets(data_root: Path) -> tuple[dict[str, Any], list[str]]:
    """Check role files and annotation paths without opening annotation data."""

    assets: dict[str, Any] = {}
    problems: list[str] = []
    annotations = {
        "train_annotation": data_root / "raw/tao/annotations/train.json",
        "val_annotation": data_root / "raw/tao/annotations/validation.json",
    }
    for role, relative in annotations.items():
        item = _asset(relative, role)
        item["content_checked"] = False
        assets[role] = item
        if not item["exists"]:
            problems.append(role + ":missing")

    for role, relative in ROLE_FILES:
        item = _asset(data_root / "tao_ow_ocd_v1" / relative, role, read_json=True, hash_file=True)
        payload = item.get("payload")
        if isinstance(payload, list):
            item["item_count"] = len(payload)
            item.pop("payload", None)
        assets[role] = item
        if not item["exists"]:
            problems.append(role + ":missing")
        elif item.get("read_error"):
            problems.append(role + ":invalid_json")
    return assets, problems


def _test_semantic_violation(value: Any, path: str = "") -> list[str]:
    """Find only explicit positive Test-semantic access flags in small JSON."""

    violations: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            lowered = str(key).lower()
            if lowered in {"test_semantic_accessed", "tao_test_used", "test_used"} and child is True:
                violations.append(child_path)
            violations.extend(_test_semantic_violation(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(_test_semantic_violation(child, f"{path}[{index}]"))
    return violations


def _stage_valid(item: dict[str, Any]) -> tuple[bool, list[str]]:
    payload = _safe_payload(item)
    problems: list[str] = []
    if not item.get("exists"):
        return False, ["covtrack_native_stage:missing"]
    if item.get("read_error"):
        return False, ["covtrack_native_stage:invalid_json"]
    if payload.get("status") != "BAKEOFF_METRICS_COMPLETE":
        problems.append("covtrack_native_stage:status_not_bakeoff_complete")
    for field in REQUIRED_STAGE_FIELDS:
        if payload.get(field) is not True:
            problems.append(f"covtrack_native_stage:{field}_not_true")
    if payload.get("test_semantic_accessed") is not False:
        problems.append("covtrack_native_stage:test_semantic_access_not_false")
    normalized = payload.get("normalized_outputs")
    if not isinstance(normalized, dict) or not normalized.get("physical_stream"):
        problems.append("covtrack_native_stage:physical_stream_path_missing")
    problems.extend(f"covtrack_native_stage:test_flag:{value}" for value in _test_semantic_violation(payload))
    return not problems, problems


def _manifest_status(manifest_item: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    payload = _safe_payload(manifest_item)
    problems: list[str] = []
    evidence: dict[str, Any] = {"shards_checked": 0, "arrays_opened": False}
    if not manifest_item.get("exists"):
        return False, ["formal_cache_manifest:missing"], evidence
    if manifest_item.get("read_error"):
        return False, ["formal_cache_manifest:invalid_json"], evidence
    if payload.get("status") != "COMPLETE":
        problems.append("formal_cache_manifest:status_not_complete")
    if payload.get("split") != "val":
        problems.append("formal_cache_manifest:split_not_val")
    if payload.get("test_semantic_accessed") is not False:
        problems.append("formal_cache_manifest:test_semantic_access_not_false")
    if payload.get("format") != "sharded_numpy_with_parquet_index":
        problems.append("formal_cache_manifest:format_mismatch")
    encoder = payload.get("encoder")
    if not isinstance(encoder, dict):
        problems.append("formal_cache_manifest:encoder_missing")
    else:
        for field in ("text_or_category_logits", "category_id_feature", "physical_id_feature", "future_observations_used"):
            if encoder.get(field) is not False:
                problems.append(f"formal_cache_manifest:encoder_{field}_not_false")
    cache_root_raw = payload.get("cache_root")
    if not cache_root_raw:
        problems.append("formal_cache_manifest:cache_root_missing")
    else:
        cache_root = Path(str(cache_root_raw))
        evidence["cache_root"] = str(cache_root)
        if "test" in {part.lower() for part in cache_root.parts}:
            problems.append("formal_cache_manifest:cache_root_is_test_path")
        completed = payload.get("completed_shards")
        if not isinstance(completed, list) or not completed:
            problems.append("formal_cache_manifest:completed_shards_missing")
        else:
            evidence["shards_declared"] = len(completed)
            for entry in completed:
                try:
                    track_start = int(entry["track_start"])
                    shard_index = track_start // int(payload["shard_tracks"])
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    problems.append("formal_cache_manifest:malformed_shard_entry")
                    continue
                shard = cache_root / "shards" / f"shard-{shard_index:06d}"
                evidence["shards_checked"] += 1
                for name in (".done", "observations.npy", "prefix_features.npy", "full_features.npy", "index.parquet"):
                    if not (shard / name).is_file():
                        problems.append(f"formal_cache_manifest:missing:{shard / name}")
                done = _asset(shard / ".done", f"shard_{shard_index}_done", read_json=True, hash_file=True)
                done_payload = _safe_payload(done)
                if done_payload:
                    done_start = _int_or_none(done_payload.get("track_start"))
                    done_count = _int_or_none(done_payload.get("track_count"))
                    expected_count = _int_or_none(entry.get("track_count"))
                    if done_start is None or done_start != track_start:
                        problems.append(f"formal_cache_manifest:done_track_start_mismatch:{shard_index}")
                    if done_count is None or expected_count is None or done_count != expected_count:
                        problems.append(f"formal_cache_manifest:done_track_count_mismatch:{shard_index}")
    problems.extend(f"formal_cache_manifest:test_flag:{value}" for value in _test_semantic_violation(payload))
    evidence["arrays_opened"] = False
    return not problems, problems, evidence


def _check_frontend_files(output_root: Path) -> tuple[dict[str, Any], list[str]]:
    assets: dict[str, Any] = {}
    problems: list[str] = []
    audit = output_root / "audit"
    for role, relative in (
        ("frontend_selection", "frontend_selection.json"),
        ("representation_decision", "representation_decision.json"),
        ("autonomous_state", "autonomous_state.json"),
        ("covtrack_native_stage", "frontend_covtrack_native.json"),
    ):
        item = _asset(audit / relative, role, read_json=True, hash_file=True)
        assets[role] = item
        if not item["exists"]:
            problems.append(role + ":missing")
        elif item.get("read_error"):
            problems.append(role + ":invalid_json")

    for role, relative in SUPPLEMENTAL_FRONTEND_FILES:
        item = _asset(audit / relative, role, read_json=True, hash_file=True)
        assets[role] = item
        # These are useful historical/provenance supplements, but the canonical
        # stage artifact remains the formal admission authority.

    selection = _safe_payload(assets["frontend_selection"])
    representation = _safe_payload(assets["representation_decision"])
    selected = selection.get("selected_frontend")
    assets["frontend_selection"]["selected_frontend"] = selected
    if selection.get("status") != "FINAL_PHYSICAL_FRONTEND_SELECTED":
        problems.append("frontend_selection:status_not_final")
    if selected not in FRONTEND_SLUGS:
        problems.append("frontend_selection:selected_frontend_invalid")
    if representation.get("status") != "FINAL_REPRESENTATION_SELECTED":
        problems.append("representation_decision:status_not_final")
    if representation.get("formal_common_feature_cache_authorized") is not True:
        problems.append("representation_decision:formal_cache_not_authorized")
    for role, item in assets.items():
        problems.extend(f"{role}:test_flag:{value}" for value in _test_semantic_violation(item.get("payload")))
    stage_ok, stage_problems = _stage_valid(assets["covtrack_native_stage"])
    assets["covtrack_native_stage"]["valid_for_admission"] = stage_ok
    problems.extend(stage_problems)
    return assets, problems


def _cache_assets(output_root: Path, selected_frontend: str | None) -> tuple[dict[str, Any], list[str], str | None]:
    manifests: dict[str, Any] = {}
    problems: list[str] = []
    for frontend, slug in FRONTEND_SLUGS.items():
        path = output_root / "features/formal" / slug / "cache_manifest.json"
        item = _asset(path, f"cache_manifest_{slug}", read_json=True, hash_file=True)
        ok, item_problems, evidence = _manifest_status(item)
        item["valid_for_admission"] = ok
        item["integrity_evidence"] = evidence
        manifests[slug] = item
        if frontend == selected_frontend:
            problems.extend(item_problems)
    selected_slug = FRONTEND_SLUGS.get(selected_frontend or "")
    if selected_slug is None:
        problems.append("formal_cache:cannot_choose_manifest_without_valid_selection")
    elif not manifests[selected_slug].get("valid_for_admission"):
        # _manifest_status problems are already included above; retain an
        # explicit stable reason for consumers that only inspect the summary.
        problems.append("formal_cache:selected_frontend_manifest_not_admissible")
    return manifests, problems, selected_slug


def build_report(project_root: Path, data_root: Path, output_root: Path) -> dict[str, Any]:
    """Build the report without writing any file or opening Test assets."""

    data_assets, data_problems = _data_assets(data_root)
    frontend_assets, frontend_problems = _check_frontend_files(output_root)
    selected = frontend_assets["frontend_selection"].get("selected_frontend")
    cache_assets, cache_problems, selected_slug = _cache_assets(output_root, selected)
    predicted = _asset(
        data_root / "tao_ow_ocd_v1/public/pred_track_stream.jsonl",
        "predicted_stream",
    )
    predicted["content_checked"] = False
    predicted["hash_skipped"] = "large_stream_not_opened"
    if not predicted["exists"]:
        data_problems.append("predicted_stream:missing")

    causal_order_assets: dict[str, Any] = {}
    public_root = data_root / "tao_ow_ocd_v1/public"
    for filename in CAUSAL_ORDER_FILES:
        item = _asset(public_root / filename, f"causal_order_{filename}")
        item["content_checked"] = False
        causal_order_assets[filename] = item
        if not item["exists"]:
            data_problems.append(f"causal_order:{filename}:missing")

    problems = data_problems + frontend_problems + cache_problems
    formal_ready = not problems
    status = "READY_FORMAL_V2_PREDICTED_EVAL_INPUTS" if formal_ready else "BLOCKED_MISSING_OR_INVALID_FORMAL_V2_ASSETS"
    return {
        "schema_version": "trackocd.v2.recovery_asset_admission.v1",
        "status": status,
        "generated_utc": _utc_now(),
        "project_root": str(project_root.resolve(strict=False)),
        "data_root": str(data_root.resolve(strict=False)),
        "output_root": str(output_root.resolve(strict=False)),
        "read_only": True,
        "network_used": False,
        "external_processes_touched": False,
        "write_performed": False,
        "test_semantic_accessed": False,
        "test_paths_opened": [],
        "formal_ready": formal_ready,
        "selected_frontend": selected,
        "selected_frontend_slug": selected_slug,
        "git": _git_info(project_root),
        "data_assets": {**data_assets, "predicted_stream": predicted, "causal_order": causal_order_assets},
        "frontend_assets": frontend_assets,
        "formal_cache_manifests": cache_assets,
        "problems": problems,
        "next_action": (
            "run the registered frozen predicted evaluator"
            if formal_ready
            else "recover/hash only the listed missing or invalid v2 assets; do not start formal evaluation"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="explicitly write the JSON report here")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    project_root = args.project_root.resolve(strict=False)
    data_root = (args.data_root or project_root / "data").resolve(strict=False)
    output_root = (args.output_root or project_root / "outputs/trackocd_v2").resolve(strict=False)
    payload = build_report(project_root, data_root, output_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload["write_performed"] = True
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["formal_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
