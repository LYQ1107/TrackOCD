#!/usr/bin/env python3
"""Replay the frozen semantic adapter plus safe controller on predicted Val.

The causal pass reads only the selected frontend's formal sharded visual
features.  The evaluator-only GT join and fixed denominator are performed
after all decisions are sealed.  This script is intentionally downstream of
the formal cache and never opens Test semantic labels.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.trackocd_v2 import run_pred_baselines as baseline  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.methods.safe_controller import SafePersistentController  # noqa: E402
from src.trackocd_v2.models.semantic_adapter import SemanticAdapter  # noqa: E402


PREFIXES = (1, 2, 4, 8, 16)
ADAPTER_SELECTION = OUTPUT_TARGET / "audit/semantic_adapter_selection.json"
SAFE_SELECTION = OUTPUT_TARGET / "audit/safe_controller_val_selection.json"
DECISION_ROOT = OUTPUT_TARGET / "diagnostics"
OUTPUT = OUTPUT_TARGET / "tables/pred_safe_controller.json"
LAUNCHED = OUTPUT_TARGET / "diagnostics/pred_safe_controller.launched"
DONE = OUTPUT_TARGET / "diagnostics/pred_safe_controller.done"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _require_lineage() -> tuple[dict[str, Any], dict[str, Any], Any]:
    if not ADAPTER_SELECTION.is_file() or not SAFE_SELECTION.is_file():
        raise RuntimeError("predicted safe controller requires adapter and Val safe-controller selections")
    adapter_selection = json.loads(ADAPTER_SELECTION.read_text(encoding="utf-8"))
    safe_selection = json.loads(SAFE_SELECTION.read_text(encoding="utf-8"))
    checkpoint = Path(str(adapter_selection.get("checkpoint", ""))).resolve()
    if adapter_selection.get("status") != "SEMANTIC_ADAPTER_SELECTED" or not checkpoint.is_file():
        raise RuntimeError("semantic adapter selection/checkpoint is incomplete")
    parameters = dict(safe_selection.get("chosen_parameters") or {})
    if not parameters:
        raise RuntimeError("safe-controller parameters are absent")
    source = baseline._resolve_causal_stream()
    if source is None:
        raise RuntimeError("formal predicted feature cache is not ready")
    return adapter_selection, safe_selection, (checkpoint, source, parameters)


def _load_model(checkpoint: Path, torch: Any) -> Any:
    payload = torch.load(checkpoint, map_location="cpu")
    model = SemanticAdapter()
    model.load_state_dict(payload["model"])
    return model.eval()


def _encode(model: Any, values: np.ndarray, torch: Any, device: Any) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(values).to(device))["z_semantic"].cpu().numpy().astype(np.float32)


def _known_prototypes(model: Any, torch: Any, device: Any) -> dict[int, dict[int, np.ndarray]]:
    manifest = _jsonl(OUTPUT_TARGET / "manifests/tao_train_gt_tracks.jsonl")
    labels = {str(row["sample_key"]): row for row in _jsonl(OUTPUT_TARGET / "manifests/private_tao_train_gt_track_labels.jsonl")}
    grouped: dict[int, defaultdict[int, list[np.ndarray]]] = {prefix: defaultdict(list) for prefix in PREFIXES}
    for row in manifest:
        key = str(row["sample_key"])
        if str(labels[key].get("gt_split")) != "old":
            continue
        payload = json.loads((OUTPUT_TARGET / "features/gt_tracks/train" / f"{key}.json").read_text(encoding="utf-8"))
        vectors = np.asarray([payload["prefix_features"][str(prefix)] for prefix in PREFIXES], dtype=np.float32)
        semantic = _encode(model, vectors, torch, device)
        category = int(labels[key]["gt_category_id"])
        for index, prefix in enumerate(PREFIXES):
            grouped[prefix][category].append(semantic[index])
    return {
        prefix: {
            category: vector / max(float(np.linalg.norm(vector)), 1e-12)
            for category, values in categories.items()
            for vector in [np.mean(values, axis=0)]
        }
        for prefix, categories in grouped.items()
    }


def _atomic_decisions(paths: dict[int, Path], stream: Path, store: Any, model: Any, prototypes: dict[int, dict[int, np.ndarray]], parameters: dict[str, Any], torch: Any, device: Any, batch_size: int) -> int:
    DECISION_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = {prefix: path.with_name(f".{path.name}.tmp.{os.getpid()}") for prefix, path in paths.items()}
    handles = {prefix: temporary[prefix].open("w", encoding="utf-8") for prefix in PREFIXES}
    controllers = {prefix: SafePersistentController(known_prototypes=prototypes[prefix], **parameters) for prefix in PREFIXES}
    batch_rows: list[dict[str, Any]] = []
    count = 0

    def flush() -> None:
        nonlocal count
        if not batch_rows:
            return
        matrices = {prefix: np.asarray([row["features"][prefix] for row in batch_rows], dtype=np.float32) for prefix in PREFIXES}
        encoded = {prefix: _encode(model, matrices[prefix], torch, device) for prefix in PREFIXES}
        for index, row in enumerate(batch_rows):
            for prefix in PREFIXES:
                decision = controllers[prefix].step(encoded[prefix][index])
                handles[prefix].write(json.dumps({"sample_key": row["sample_key"], "stream_order": row["stream_order"], "decision": decision}, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
        batch_rows.clear()

    try:
        for row in baseline._read_jsonl(stream):
            key = str(row["sample_key"])
            feature = store.get(key)
            batch_rows.append({"sample_key": key, "stream_order": int(row["stream_order"]), "features": {prefix: np.asarray(feature["prefix_features"][str(prefix)], dtype=np.float32) for prefix in PREFIXES}})
            if len(batch_rows) >= batch_size:
                flush()
        flush()
    finally:
        for handle in handles.values():
            handle.close()
    for prefix in PREFIXES:
        os.replace(temporary[prefix], paths[prefix])
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = ensure_output_layout()
    if DONE.exists() and not args.resume:
        print(json.dumps({"status": "ALREADY_DONE", "done": str(DONE.resolve())}, indent=2))
        return 0
    if LAUNCHED.exists() and not args.resume:
        try:
            marker = json.loads(LAUNCHED.read_text(encoding="utf-8"))
            os.kill(int(marker["pid"]), 0)
        except (OSError, ProcessLookupError):
            raise RuntimeError(f"stale launch marker requires explicit --resume: {LAUNCHED}")
        raise RuntimeError(f"predicted safe controller is already owned by PID {marker['pid']}")

    import torch

    torch.set_num_threads(1)
    adapter_selection, safe_selection, (checkpoint, source, parameters) = _require_lineage()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = _load_model(checkpoint, torch).to(device)
    if not LAUNCHED.exists():
        atomic_json(LAUNCHED, {"pid": os.getpid(), "started_utc": _now(), "device": str(device), "source": source, "adapter_selection_sha256": sha256_file(ADAPTER_SELECTION), "safe_selection_sha256": sha256_file(SAFE_SELECTION)})
    prototypes = _known_prototypes(model, torch, device)
    stream, store, source_meta = source
    paths = {prefix: DECISION_ROOT / f"{source_meta['decision_namespace']}_safe_controller_decisions_p{prefix}.jsonl" for prefix in PREFIXES}
    decision_count = _atomic_decisions(paths, stream, store, model, prototypes, parameters, torch, device, max(1, int(args.batch_size)))
    join_audit, join_path = baseline._ensure_evaluator_join(source_meta)
    join_rows = baseline._load_join(join_path)
    gt_rows = baseline._load_fixed_gt_rows()
    known = baseline.load_ids(baseline.ROLE_ROOT / "known_ids.json")
    novel = baseline.load_ids(baseline.ROLE_ROOT / "unknown_ids_val.json")
    distractor = baseline.load_ids(baseline.ROLE_ROOT / "distractor_ids.json")
    metrics, missing = baseline._evaluate(paths, join_rows, gt_rows, known, novel, distractor)
    aggregate = {
        prefix: {
            "standard_mean": {name: float(metrics[prefix]["standard"][name]) for name in ("old_acc", "new_acc", "h_score", "all_acc")},
            "persistent_mean": {name: float(metrics[prefix]["persistent"][name]) for name in ("commit_ct", "false_assignment_rate")},
        }
        for prefix in map(str, PREFIXES)
    }
    result = {
        "schema_version": "trackocd.v2.pred_safe_controller.v1",
        "status": "COMPLETE",
        "generated_utc": _now(),
        "method": "SemanticAdapter+SafePersistentController",
        "representation": "selected frontend formal sharded visual feature -> z_semantic",
        "decision_count": decision_count,
        "aggregate": aggregate,
        "prefixes": metrics,
        "fixed_gt_denominator_rows": len(gt_rows),
        "unmatched_gt_targets_counted_as_defer": missing,
        "fixed_gt_denominator": True,
        "source": source_meta,
        "adapter_selection": str(ADAPTER_SELECTION.resolve()),
        "adapter_selection_sha256": sha256_file(ADAPTER_SELECTION),
        "safe_controller_selection": str(SAFE_SELECTION.resolve()),
        "safe_controller_selection_sha256": sha256_file(SAFE_SELECTION),
        "evaluator_join": str(join_path.resolve()),
        "evaluator_join_sha256": sha256_file(join_path),
        "evaluator_join_audit": str((OUTPUT_TARGET / "audit" / f"frontend_{source_meta['frontend_slug']}_evaluator_join.json").resolve()),
        "test_semantic_accessed": False,
    }
    atomic_json(OUTPUT, result)
    atomic_json(DONE, {"pid": os.getpid(), "decision_count": decision_count, "completed_utc": _now(), "output": str(OUTPUT.resolve())})
    if LAUNCHED.exists():
        LAUNCHED.unlink()
    print(json.dumps({"status": result["status"], "method": result["method"], "decision_count": decision_count, "aggregate_p16": aggregate["16"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
