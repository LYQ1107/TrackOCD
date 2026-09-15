#!/usr/bin/env python3
"""Run fixed baselines on the complete public predicted-track stream.

The causal pass iterates over every public predicted track and reads only its
v2 visual feature.  The evaluator-only temporal-IoU join is built and used
only after the public decisions have been sealed in per-prefix JSONL files.
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

from src.trackocd_v2.evaluation.persistent import evaluate_persistent  # noqa: E402
from src.trackocd_v2.evaluation.standard_ocd import evaluate_standard  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.methods.dpmeans import OnlineDPMeans  # noqa: E402
from src.trackocd_v2.methods.nearest_prototype import NearestPrototype  # noqa: E402
from src.trackocd_v2.methods.phe_track import PHETrackAdapter  # noqa: E402
from src.trackocd_v2.protocol import load_ids  # noqa: E402


PREFIXES = (1, 2, 4, 8, 16)
MANIFEST = OUTPUT_TARGET / "manifests/tao_val_predicted_tracks.jsonl"
TRAIN_MANIFEST = OUTPUT_TARGET / "manifests/tao_train_gt_tracks.jsonl"
TRAIN_LABELS = OUTPUT_TARGET / "manifests/private_tao_train_gt_track_labels.jsonl"
FEATURE_ROOT = OUTPUT_TARGET / "features/pred_tracks/pred"
TRAIN_FEATURE_ROOT = OUTPUT_TARGET / "features/gt_tracks/train"
FEATURE_AUDIT = OUTPUT_TARGET / "audit/predicted_feature_audit.json"
JOIN_AUDIT = OUTPUT_TARGET / "audit/predicted_evaluator_join.json"
JOIN_PATH = OUTPUT_TARGET / "manifests/tao_val_predicted_evaluator_join.jsonl"
DECISION_ROOT = OUTPUT_TARGET / "diagnostics"
TABLE_ROOT = OUTPUT_TARGET / "tables"
ROLE_ROOT = ROOT / "data/tao_ow_ocd_v1/splits"
PHE_CHECKPOINT = ROOT / "runs/phe_track/dinov2_seed1027/checkpoint.pth"


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else np.zeros_like(value)


def _load_prototypes() -> dict[int, dict[int, np.ndarray]]:
    labels = {str(row["sample_key"]): row for row in _read_jsonl(TRAIN_LABELS)}
    groups: dict[int, defaultdict[int, list[np.ndarray]]] = {
        prefix: defaultdict(list) for prefix in PREFIXES
    }
    for row in _read_jsonl(TRAIN_MANIFEST):
        key = str(row["sample_key"])
        label = labels.get(key)
        if label is None:
            raise ValueError(f"training label sidecar missing {key}")
        if str(label["gt_split"]) != "old":
            continue
        feature_path = TRAIN_FEATURE_ROOT / f"{key}.json"
        feature = json.loads(feature_path.read_text(encoding="utf-8"))
        for prefix in PREFIXES:
            vector = np.asarray(feature["prefix_features"][str(prefix)], dtype=np.float32)
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError(f"invalid train prototype feature for {key}, p{prefix}")
            groups[prefix][int(label["gt_category_id"])].append(vector)
    return {
        prefix: {category: _unit(np.mean(values, axis=0)) for category, values in by_category.items()}
        for prefix, by_category in groups.items()
    }


def _make_methods(method_name: str, prototypes: dict[int, dict[int, np.ndarray]], radius: int, device: str) -> dict[int, Any]:
    if method_name == "nearest":
        return {prefix: NearestPrototype(known_prototypes=prototypes[prefix]) for prefix in PREFIXES}
    if method_name == "dpmeans":
        return {prefix: OnlineDPMeans(known_prototypes=prototypes[prefix]) for prefix in PREFIXES}
    if method_name == "phe":
        if not PHE_CHECKPOINT.exists():
            raise FileNotFoundError(PHE_CHECKPOINT)
        return {prefix: PHETrackAdapter(PHE_CHECKPOINT, radius=radius, device=device) for prefix in PREFIXES}
    raise ValueError(method_name)


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def _public_decisions(method_name: str, methods: dict[int, Any]) -> tuple[dict[int, Path], int]:
    DECISION_ROOT.mkdir(parents=True, exist_ok=True)
    final_paths = {prefix: DECISION_ROOT / f"pred_{method_name}_decisions_p{prefix}.jsonl" for prefix in PREFIXES}
    temporary_paths = {prefix: _temporary(path) for prefix, path in final_paths.items()}
    handles = {}
    count = 0
    previous_order = -1
    try:
        handles = {prefix: path.open("w", encoding="utf-8") for prefix, path in temporary_paths.items()}
        for row in _read_jsonl(MANIFEST):
            key = str(row["sample_key"])
            stream_order = int(row["stream_order"])
            if stream_order < previous_order:
                raise ValueError(f"public stream order regressed at {key}: {stream_order} < {previous_order}")
            previous_order = stream_order
            feature_path = FEATURE_ROOT / f"{key}.json"
            if not feature_path.exists():
                raise FileNotFoundError(feature_path)
            feature = json.loads(feature_path.read_text(encoding="utf-8"))
            if str(feature.get("source_split")) != "pred":
                raise ValueError(f"predicted decision read non-predicted feature: {feature_path}")
            for prefix in PREFIXES:
                vector = np.asarray(feature["prefix_features"][str(prefix)], dtype=np.float32)
                if vector.shape != (768,) or not np.isfinite(vector).all():
                    raise ValueError(f"invalid predicted feature for {key}, p{prefix}")
                decision = dict(methods[prefix].step(vector))
                handles[prefix].write(json.dumps({
                    "sample_key": key,
                    "stream_order": stream_order,
                    "decision": decision,
                }, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    finally:
        for handle in handles.values():
            handle.close()
    for prefix in PREFIXES:
        os.replace(temporary_paths[prefix], final_paths[prefix])
    return final_paths, count


def _ensure_evaluator_join() -> dict:
    if JOIN_AUDIT.exists() and JOIN_PATH.exists():
        audit = json.loads(JOIN_AUDIT.read_text(encoding="utf-8"))
        if (
            audit.get("status") == "COMPLETE"
            and audit.get("public_prediction_source_sha256") == sha256_file(ROOT / "data/tao_ow_ocd_v1/public/pred_track_stream.jsonl")
        ):
            return audit
    command = [sys.executable, str(ROOT / "scripts/trackocd_v2/build_predicted_evaluator_join.py")]
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"evaluator-only predicted join failed: {result.returncode}")
    return json.loads(JOIN_AUDIT.read_text(encoding="utf-8"))


def _load_join() -> list[dict]:
    return list(_read_jsonl(JOIN_PATH))


def _load_matched_decisions(path: Path, wanted: set[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in _read_jsonl(path):
        key = str(row["sample_key"])
        if key in wanted:
            if key in result:
                raise ValueError(f"duplicate decision for {key}: {path}")
            result[key] = row["decision"]
    missing = wanted - result.keys()
    if missing:
        raise ValueError(f"decision file misses {len(missing)} evaluator keys: {sorted(missing)[:5]}")
    return result


def _slim_standard(metric: dict) -> dict:
    keep = ("old_acc", "new_acc", "h_score", "all_acc", "old_correct", "old_total", "new_correct", "new_total", "state_count", "hungarian_mapping")
    return {key: metric[key] for key in keep}


def _evaluate(decision_paths: dict[int, Path], join_rows: list[dict], known_ids: set[int], novel_ids: set[int], distractor_ids: set[int]) -> dict:
    evaluator_rows = [dict(row) for row in join_rows]
    wanted = {str(row["sample_key"]) for row in evaluator_rows}
    result: dict[str, dict] = {}
    for prefix in PREFIXES:
        decision_by_key = _load_matched_decisions(decision_paths[prefix], wanted)
        decisions = [decision_by_key[str(row["sample_key"])] for row in evaluator_rows]
        standard = evaluate_standard(
            evaluator_rows,
            decisions,
            known_ids=known_ids,
            novel_ids=novel_ids,
            distractor_ids=distractor_ids,
        )
        persistent = evaluate_persistent(
            evaluator_rows,
            decisions,
            known_ids=known_ids,
            novel_ids=novel_ids,
            distractor_ids=distractor_ids,
        )
        result[str(prefix)] = {
            "standard": _slim_standard(standard),
            "persistent": persistent,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("nearest", "dpmeans", "phe"), required=True)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    out = ensure_output_layout()
    table_path = TABLE_ROOT / f"pred_{args.method}.json"
    if not FEATURE_AUDIT.exists() or json.loads(FEATURE_AUDIT.read_text(encoding="utf-8")).get("status") != "READY":
        atomic_json(table_path, {
            "schema_version": "trackocd.v2.pred_baseline.v1",
            "status": "WAITING_FEATURES",
            "method": args.method,
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        })
        return 2
    for path in (MANIFEST, TRAIN_MANIFEST, TRAIN_LABELS):
        if not path.exists():
            raise FileNotFoundError(path)

    prototypes = _load_prototypes()
    methods = _make_methods(args.method, prototypes, args.radius, args.device)
    decision_paths, decision_count = _public_decisions(args.method, methods)
    join_audit = _ensure_evaluator_join()
    join_rows = _load_join()
    known_ids = load_ids(ROLE_ROOT / "known_ids.json")
    novel_ids = load_ids(ROLE_ROOT / "unknown_ids_val.json")
    distractor_ids = load_ids(ROLE_ROOT / "distractor_ids.json")
    prefixes = _evaluate(decision_paths, join_rows, known_ids, novel_ids, distractor_ids)
    aggregate = {
        prefix: {
            "standard_mean": {
                name: float(prefixes[prefix]["standard"][name])
                for name in ("old_acc", "new_acc", "h_score", "all_acc")
            },
            "persistent_mean": {
                name: float(prefixes[prefix]["persistent"][name])
                for name in ("commit_ct", "false_assignment_rate")
            },
        }
        for prefix in prefixes
    }
    result = {
        "schema_version": "trackocd.v2.pred_baseline.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": args.method,
        "prefixes": list(PREFIXES),
        "fixed_method_parameters": {
            "nearest": {"tau_known": 0.35, "tau_existing": 0.55},
            "dpmeans": {"lambda_distance": 0.45, "tau_known": 0.35},
            "phe": {"radius": args.radius, "checkpoint": str(PHE_CHECKPOINT.resolve())},
        }[args.method],
        "train_known_prototype_categories_by_prefix": {
            str(prefix): sorted(int(key) for key in prototypes[prefix]) for prefix in PREFIXES
        },
        "public_stream": {
            "manifest": str(MANIFEST.resolve()),
            "manifest_sha256": sha256_file(MANIFEST),
            "decision_count": decision_count,
            "decision_files": {str(prefix): str(path.resolve()) for prefix, path in decision_paths.items()},
            "all_public_tracks_received_a_causal_decision": True,
        },
        "evaluator_subset": {
            "join_audit": str(JOIN_AUDIT.resolve()),
            "join_audit_sha256": sha256_file(JOIN_AUDIT),
            "join_rows": len(join_rows),
            "public_decisions_filtered_only_after_causal_pass": True,
            "gt_matching_used_as_model_input": False,
            "historical_matched_stream_used_for_inference": False,
            "matching_source": join_audit.get("historical_diagnostic_source"),
        },
        "prefixes_result": prefixes,
        "aggregate": aggregate,
        "test_semantic_accessed": False,
        "model_input_contract": {
            "input_fields": ["DINOv2 prefix feature", "public stream order"],
            "gt_category_id": False,
            "gt_split": False,
            "temporal_iou_match": False,
            "text_or_category_logits": False,
        },
    }
    atomic_json(table_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
