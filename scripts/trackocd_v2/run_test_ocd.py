#!/usr/bin/env python3
"""Run the frozen TrackOCD v2 OCD evaluation on TAO Test.

The runner has one deliberate phase boundary:

1. Read only the selected frontend's category-free Test stream and sharded
   visual cache, then seal all causal decisions.
2. Materialize the final Test GT manifest, perform the evaluator-only temporal
   join, and score Standard/Persistent OCD.

No Test category or role is loaded during phase 1.  The immutable
``FINAL_FREEZE.json`` is required before phase 2 and prevents any Test-side
selection or threshold tuning.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.trackocd_v2 import run_pred_baselines as baseline  # noqa: E402
from scripts.trackocd_v2 import run_pred_safe_controller as safe_runner  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.methods.safe_controller import SafePersistentController  # noqa: E402
from src.trackocd_v2.protocol import assert_test_semantic_access_allowed, load_ids  # noqa: E402
from src.trackocd_v2.sharded_cache import ShardedFeatureStore  # noqa: E402


PREFIXES = (1, 2, 4, 8, 16)
FRONTEND_NAMES = {
    "SimOWT/Q0": "simowt",
    "OVTR-native": "ovtr",
    "COVTrack-native": "covtrack_native",
    "COVTrack-NoSemantic": "covtrack_nosem",
}
ROLE_ROOT = ROOT / "data/tao_ow_ocd_v1/splits"
MANIFEST_ROOT = OUTPUT_TARGET / "manifests"
FEATURE_ROOT = OUTPUT_TARGET / "features/formal"
FINAL_FREEZE = OUTPUT_TARGET / "audit/FINAL_FREEZE.json"
ADAPTER_SELECTION = OUTPUT_TARGET / "audit/semantic_adapter_selection.json"
SAFE_SELECTION = OUTPUT_TARGET / "audit/safe_controller_val_selection.json"
DECISION_ROOT = OUTPUT_TARGET / "diagnostics"
TABLE_ROOT = OUTPUT_TARGET / "tables"
OUTPUT = TABLE_ROOT / "test_ocd.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_number} is not an object: {path}")
                yield value


def _selected_source(
    *,
    physical_stream: Path | None,
    cache_manifest: Path | None,
) -> tuple[str, Path, ShardedFeatureStore, dict[str, Any]]:
    """Resolve only post-freeze physical/cache lineage, without Test labels."""

    assert_test_semantic_access_allowed(FINAL_FREEZE, "resolve frozen TAO Test OCD source")
    freeze = json.loads(FINAL_FREEZE.read_text(encoding="utf-8"))
    frontend = str(freeze.get("frontend") or "")
    slug = FRONTEND_NAMES.get(frontend)
    if slug is None:
        raise RuntimeError(f"FINAL_FREEZE has an unknown frontend: {frontend}")
    stage_path = OUTPUT_TARGET / "audit" / f"frontend_{slug}_test.json"
    if not stage_path.is_file():
        raise RuntimeError(f"frozen Test frontend stream is missing: {stage_path}")
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if stage.get("status") != "FINAL_TEST_NATIVE_STREAM_NORMALIZED":
        raise RuntimeError(f"frozen Test frontend stream is incomplete: {stage.get('status')}")
    normalized = stage.get("normalized_outputs") or {}
    default_stream = Path(str(normalized.get("physical_stream", ""))).resolve()
    stream = (physical_stream or default_stream).resolve()
    if stream != default_stream or not stream.is_file():
        raise RuntimeError("Test OCD input must be the frozen selected frontend physical stream")
    default_cache = (FEATURE_ROOT / slug / "test" / "cache_manifest.json").resolve()
    manifest_path = (cache_manifest or default_cache).resolve()
    if manifest_path != default_cache or not manifest_path.is_file():
        raise RuntimeError("Test OCD input must be the frozen selected frontend test cache")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("split") != "test" or manifest.get("frontend") != frontend:
        raise RuntimeError("selected frontend Test cache is not complete or lineage-aligned")
    store = ShardedFeatureStore(manifest_path)
    return frontend, stream, store, {
        "frontend_slug": slug,
        "frontend_stage": str(stage_path.resolve()),
        "frontend_stage_sha256": sha256_file(stage_path),
        "physical_stream": str(stream),
        "physical_stream_sha256": sha256_file(stream),
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": sha256_file(manifest_path),
    }


def _validate_feature(feature: dict[str, Any], key: str) -> None:
    forbidden = {"category_id", "category_name", "text", "gt_category_id", "gt_split", "gt_match_id"}
    if forbidden & set(feature):
        raise ValueError(f"Test causal feature contains semantic/evaluator fields: {key}")
    if str(feature.get("source_split")) != "test_predicted":
        raise ValueError(f"Test causal feature has unexpected source split: {key}")


def _atomic_decisions(
    *,
    method_name: str,
    stream: Path,
    store: ShardedFeatureStore,
    prototypes: dict[int, dict[int, np.ndarray]],
    radius: int,
    device: str,
) -> tuple[dict[int, Path], int]:
    methods = baseline._make_methods(method_name, prototypes, radius, device)
    slug = str(json.loads(FINAL_FREEZE.read_text(encoding="utf-8"))["frontend"])
    slug = FRONTEND_NAMES[slug]
    DECISION_ROOT.mkdir(parents=True, exist_ok=True)
    final_paths = {
        prefix: DECISION_ROOT / f"test_{slug}_{method_name}_decisions_p{prefix}.jsonl"
        for prefix in PREFIXES
    }
    temporary_paths = {
        prefix: path.with_name(f".{path.name}.tmp.{os.getpid()}")
        for prefix, path in final_paths.items()
    }
    handles = {prefix: path.open("w", encoding="utf-8") for prefix, path in temporary_paths.items()}
    seen: set[str] = set()
    previous_order = -1
    count = 0
    try:
        for row in _jsonl(stream):
            key = str(row["sample_key"])
            if key in seen:
                raise ValueError(f"duplicate Test physical sample key: {key}")
            seen.add(key)
            order = int(row["stream_order"])
            if order < previous_order:
                raise ValueError(f"Test causal stream order regressed at {key}")
            previous_order = order
            feature = store.get(key)
            _validate_feature(feature, key)
            for prefix in PREFIXES:
                vector = np.asarray(feature["prefix_features"][str(prefix)], dtype=np.float32)
                if vector.shape != (768,) or not np.isfinite(vector).all():
                    raise ValueError(f"invalid Test feature for {key}, p{prefix}")
                decision = dict(methods[prefix].step(vector))
                handles[prefix].write(json.dumps({
                    "sample_key": key,
                    "stream_order": order,
                    "decision": decision,
                }, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    finally:
        for handle in handles.values():
            handle.close()
    for prefix in PREFIXES:
        os.replace(temporary_paths[prefix], final_paths[prefix])
    return final_paths, count


def _atomic_safe_decisions(
    *,
    stream: Path,
    store: ShardedFeatureStore,
    batch_size: int,
    device_name: str,
) -> tuple[dict[int, Path], int]:
    if not ADAPTER_SELECTION.is_file() or not SAFE_SELECTION.is_file():
        raise RuntimeError("frozen Test safe-controller route requires adapter and Val controller selections")
    adapter_selection = json.loads(ADAPTER_SELECTION.read_text(encoding="utf-8"))
    safe_selection = json.loads(SAFE_SELECTION.read_text(encoding="utf-8"))
    checkpoint = Path(str(adapter_selection.get("checkpoint", ""))).resolve()
    parameters = dict(safe_selection.get("chosen_parameters") or {})
    if adapter_selection.get("status") != "SEMANTIC_ADAPTER_SELECTED" or not checkpoint.is_file():
        raise RuntimeError("semantic adapter checkpoint is incomplete")
    if not parameters:
        raise RuntimeError("Val safe-controller parameters are missing")

    import torch

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = safe_runner._load_model(checkpoint, torch).to(device)
    prototypes = safe_runner._known_prototypes(model, torch, device)
    slug = FRONTEND_NAMES[str(json.loads(FINAL_FREEZE.read_text(encoding="utf-8"))["frontend"])]
    DECISION_ROOT.mkdir(parents=True, exist_ok=True)
    final_paths = {
        prefix: DECISION_ROOT / f"test_{slug}_safe_controller_decisions_p{prefix}.jsonl"
        for prefix in PREFIXES
    }
    temporary_paths = {
        prefix: path.with_name(f".{path.name}.tmp.{os.getpid()}")
        for prefix, path in final_paths.items()
    }
    handles = {prefix: path.open("w", encoding="utf-8") for prefix, path in temporary_paths.items()}
    controllers = {
        prefix: SafePersistentController(known_prototypes=prototypes[prefix], **parameters)
        for prefix in PREFIXES
    }
    pending: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_order = -1
    count = 0

    def encode(values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(values).to(device))["z_semantic"].cpu().numpy().astype(np.float32)

    def flush() -> None:
        nonlocal count
        if not pending:
            return
        encoded = {
            prefix: encode(np.asarray([item["features"][prefix] for item in pending], dtype=np.float32))
            for prefix in PREFIXES
        }
        for index, item in enumerate(pending):
            for prefix in PREFIXES:
                decision = controllers[prefix].step(
                    encoded[prefix][index],
                    observations=int(item["observations"][prefix]),
                    quality=float(item["quality"]),
                )
                handles[prefix].write(json.dumps({
                    "sample_key": item["sample_key"],
                    "stream_order": item["stream_order"],
                    "decision": decision,
                }, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
        pending.clear()

    try:
        for row in _jsonl(stream):
            key = str(row["sample_key"])
            if key in seen:
                raise ValueError(f"duplicate Test physical sample key: {key}")
            seen.add(key)
            order = int(row["stream_order"])
            if order < previous_order:
                raise ValueError(f"Test causal stream order regressed at {key}")
            previous_order = order
            feature = store.get(key)
            _validate_feature(feature, key)
            quality = np.asarray(feature.get("quality", []), dtype=np.float32)
            if quality.ndim != 1 or not len(quality) or not np.isfinite(quality).all():
                raise ValueError(f"invalid Test feature quality for {key}")
            pending.append({
                "sample_key": key,
                "stream_order": order,
                "features": {prefix: np.asarray(feature["prefix_features"][str(prefix)], dtype=np.float32) for prefix in PREFIXES},
                "observations": {prefix: int(feature["prefix_observations_used"][str(prefix)]) for prefix in PREFIXES},
                "quality": float(np.mean(quality)),
            })
            if len(pending) >= max(1, int(batch_size)):
                flush()
        flush()
    finally:
        for handle in handles.values():
            handle.close()
    for prefix in PREFIXES:
        os.replace(temporary_paths[prefix], final_paths[prefix])
    return final_paths, count


def _load_test_gt_rows() -> list[dict[str, Any]]:
    """Load Test labels only after every causal decision is on disk."""

    assert_test_semantic_access_allowed(FINAL_FREEZE, "load frozen Test GT OCD denominator")
    manifest = MANIFEST_ROOT / "tao_test_gt_tracks.jsonl"
    labels_path = MANIFEST_ROOT / "private_tao_test_gt_track_labels.jsonl"
    labels = {str(row["sample_key"]): row for row in _jsonl(labels_path)}
    rows = []
    for row in _jsonl(manifest):
        key = str(row["sample_key"])
        label = labels.get(key)
        if label is None:
            raise ValueError(f"Test GT label sidecar missing {key}")
        if str(label.get("gt_split")) == "distractor" or bool(label.get("is_distractor", False)):
            continue
        rows.append({
            "gt_sample_key": key,
            "gt_physical_track_id": str(row["physical_track_id"]),
            "video_id": int(row["video_id"]),
            "gt_stream_order": int(row.get("stream_order", len(rows))),
            "gt_category_id": int(label["gt_category_id"]),
            "gt_split": str(label["gt_split"]),
            "evaluator_track_key": key,
        })
    if not rows:
        raise ValueError("frozen Test GT denominator is empty")
    return rows


def _aggregate(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        prefix: {
            "standard": {
                key: float(metrics[prefix]["standard"][key])
                for key in ("old_acc", "new_acc", "h_score", "all_acc")
            },
            "persistent": {
                key: float(metrics[prefix]["persistent"][key])
                for key in ("commit_ct", "false_assignment_rate")
            },
        }
        for prefix in map(str, PREFIXES)
    }


def run(
    *,
    device: str,
    batch_size: int,
    radius: int,
    methods: tuple[str, ...],
    physical_stream: Path | None = None,
    cache_manifest: Path | None = None,
) -> dict[str, Any]:
    frontend, stream, store, source = _selected_source(
        physical_stream=physical_stream,
        cache_manifest=cache_manifest,
    )

    # Phase 1: all method decisions are made without loading Test GT labels.
    prototypes = baseline._load_prototypes()
    decision_results: dict[str, dict[str, Any]] = {}
    decision_paths: dict[str, dict[int, Path]] = {}
    for method in methods:
        if method == "safe_controller":
            paths, count = _atomic_safe_decisions(
                stream=stream,
                store=store,
                batch_size=batch_size,
                device_name=device,
            )
        elif method in {"nearest", "dpmeans", "phe"}:
            paths, count = _atomic_decisions(
                method_name=method,
                stream=stream,
                store=store,
                prototypes=prototypes,
                radius=radius,
                device=device,
            )
        else:
            raise ValueError(f"unsupported frozen Test method: {method}")
        decision_paths[method] = paths
        decision_results[method] = {
            "status": "DECISIONS_SEALED",
            "decision_count": count,
            "decision_files": {str(prefix): str(path.resolve()) for prefix, path in paths.items()},
        }

    # Phase 2 starts only after every selected method has sealed its causal
    # files.  The final labelled manifest is generated at this boundary.
    build_manifest = subprocess.run(
        [sys.executable, str(ROOT / "scripts/trackocd_v2/build_test_track_stream.py"), "--final"],
        cwd=ROOT,
        check=False,
    )
    if build_manifest.returncode != 0:
        raise RuntimeError(f"final Test manifest build failed: {build_manifest.returncode}")
    gt_rows = _load_test_gt_rows()
    slug = source["frontend_slug"]
    join_path = MANIFEST_ROOT / f"frontend_{slug}_test_evaluator_join.jsonl"
    join_audit_path = OUTPUT_TARGET / "audit" / f"frontend_{slug}_test_evaluator_join.json"
    join_module = __import__("scripts.trackocd_v2.build_frontend_evaluator_join", fromlist=["run"])
    join_audit = join_module.run(
        physical_stream=stream,
        frontend=slug,
        join_path=join_path.resolve(),
        audit_path=join_audit_path.resolve(),
        split="test",
        gt_manifest=(MANIFEST_ROOT / "tao_test_gt_tracks.jsonl").resolve(),
        gt_labels=(MANIFEST_ROOT / "private_tao_test_gt_track_labels.jsonl").resolve(),
    )
    join_rows = list(_jsonl(join_path))
    gt_evaluator_rows = gt_rows
    known = load_ids(ROLE_ROOT / "known_ids.json")
    novel = load_ids(ROLE_ROOT / "unknown_ids_val.json")
    distractor = load_ids(ROLE_ROOT / "distractor_ids.json")
    for method in methods:
        metrics, missing = baseline._evaluate(
            decision_paths[method],
            join_rows,
            gt_evaluator_rows,
            known,
            novel,
            distractor,
        )
        decision_results[method].update({
            "status": "COMPLETE",
            "aggregate": _aggregate(metrics),
            "prefixes": metrics,
            "fixed_gt_denominator_rows": len(gt_rows),
            "unmatched_gt_targets_counted_as_defer": missing,
            "fixed_gt_denominator": True,
        })
    freeze_payload = json.loads(FINAL_FREEZE.read_text(encoding="utf-8"))
    controller_selection = (freeze_payload.get("controller") or {}).get("selection")
    if controller_selection:
        controller_selection_path = Path(str(controller_selection)).resolve()
        if controller_selection_path.is_file():
            controller_selection_payload = json.loads(controller_selection_path.read_text(encoding="utf-8"))
            selected_method = controller_selection_payload.get("selected_method")
        else:
            selected_method = None
    else:
        selected_method = None
    return {
        "schema_version": "trackocd.v2.test_ocd.v1",
        "status": "COMPLETE",
        "generated_utc": _now(),
        "split": "test",
        "frontend": frontend,
        "frontend_slug": slug,
        "methods": decision_results,
        "method_order": list(methods),
        "source": source,
        "test_gt_manifest": str((MANIFEST_ROOT / "tao_test_gt_tracks.jsonl").resolve()),
        "test_gt_manifest_sha256": sha256_file(MANIFEST_ROOT / "tao_test_gt_tracks.jsonl"),
        "test_gt_labels": str((MANIFEST_ROOT / "private_tao_test_gt_track_labels.jsonl").resolve()),
        "test_gt_labels_sha256": sha256_file(MANIFEST_ROOT / "private_tao_test_gt_track_labels.jsonl"),
        "evaluator_join": str(join_path.resolve()),
        "evaluator_join_sha256": sha256_file(join_path),
        "evaluator_join_audit": str(join_audit_path.resolve()),
        "evaluator_join_audit_sha256": sha256_file(join_audit_path),
        "join_summary": {
            "matched_evaluator_tracks": join_audit.get("matched_evaluator_tracks"),
            "gt_non_distractor_tracks": join_audit.get("gt_non_distractor_tracks"),
            "physical_stream_tracks": join_audit.get("physical_stream_tracks"),
            "gt_join_used_for_model_or_native_stream": join_audit.get("gt_join_used_for_model_or_native_stream"),
        },
        "final_freeze": str(FINAL_FREEZE.resolve()),
        "final_freeze_sha256": sha256_file(FINAL_FREEZE),
        "test_selection_or_tuning": False,
        "test_semantic_accessed": True,
        "causal_decisions_sealed_before_test_gt_join": True,
        "selected_method_at_freeze": selected_method,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--methods", default="nearest,dpmeans,phe,safe_controller")
    parser.add_argument("--physical-stream", type=Path, default=None)
    parser.add_argument("--cache-manifest", type=Path, default=None)
    args = parser.parse_args()
    out = ensure_output_layout()
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    try:
        result = run(
            device=args.device,
            batch_size=max(1, int(args.batch_size)),
            radius=max(1, int(args.radius)),
            methods=methods,
            physical_stream=args.physical_stream,
            cache_manifest=args.cache_manifest,
        )
    except Exception as exc:
        failure = {
            "schema_version": "trackocd.v2.test_ocd.v1",
            "status": "FAILED_TEST_OCD",
            "generated_utc": _now(),
            "error": f"{type(exc).__name__}: {exc}",
            "test_selection_or_tuning": False,
            "test_semantic_accessed": False,
        }
        atomic_json(OUTPUT, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    atomic_json(OUTPUT, result)
    print(json.dumps({
        "status": result["status"],
        "frontend": result["frontend"],
        "methods": result["method_order"],
        "test_semantic_accessed": result["test_semantic_accessed"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
