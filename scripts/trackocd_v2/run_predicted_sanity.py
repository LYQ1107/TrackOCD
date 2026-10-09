#!/usr/bin/env python3
"""Run the first predicted-track sanity benchmark with one representation.

This route deliberately uses only the already available ``full`` descriptor.
It makes causal decisions in public stream order for the bounded historical
evaluator-diagnostic candidate set, then performs the evaluator-only geometry
join.  The current cache is incomplete and the candidate membership is
evaluator-selected, so the output is explicitly diagnostic and cannot be used
as the formal predicted benchmark.
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


MANIFEST_ROOT = OUTPUT_TARGET / "manifests"
PUBLIC_MANIFEST = MANIFEST_ROOT / "tao_val_predicted_tracks.jsonl"
PRED_DIAGNOSTIC = ROOT / "data/tao_ow_ocd_v1/public/pred_track_stream_matched_iou0.5.jsonl"
TRAIN_MANIFEST = MANIFEST_ROOT / "tao_train_gt_tracks.jsonl"
TRAIN_LABELS = MANIFEST_ROOT / "private_tao_train_gt_track_labels.jsonl"
VAL_LABELS = MANIFEST_ROOT / "private_tao_val_gt_track_labels.jsonl"
FEATURE_ROOT = OUTPUT_TARGET / "features/pred_tracks/pred"
TRAIN_FEATURE_ROOT = OUTPUT_TARGET / "features/gt_tracks/train"
JOIN_SCRIPT = ROOT / "scripts/trackocd_v2/build_predicted_evaluator_join.py"
JOIN_PATH = OUTPUT_TARGET / "manifests/tao_val_predicted_evaluator_join.jsonl"
JOIN_AUDIT = OUTPUT_TARGET / "audit/predicted_evaluator_join.json"
OUTPUT = OUTPUT_TARGET / "tables/predicted_sanity_benchmark.json"
DECISION_ROOT = OUTPUT_TARGET / "diagnostics"
ROLE_ROOT = ROOT / "data/tao_ow_ocd_v1/splits"
PHE_CHECKPOINT = ROOT / "runs/phe_track/dinov2_seed1027/checkpoint.pth"
PUBLIC_SOURCE = ROOT / "data/tao_ow_ocd_v1/public/pred_track_stream.jsonl"


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else np.zeros_like(value)


def _load_prototypes() -> dict[int, np.ndarray]:
    labels = {str(row["sample_key"]): row for row in _read_jsonl(TRAIN_LABELS)}
    groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in _read_jsonl(TRAIN_MANIFEST):
        key = str(row["sample_key"])
        label = labels.get(key)
        if label is None:
            raise ValueError(f"training label sidecar missing {key}")
        if str(label["gt_split"]) != "old":
            continue
        feature_path = TRAIN_FEATURE_ROOT / f"{key}.json"
        feature = json.loads(feature_path.read_text(encoding="utf-8"))
        vector = np.asarray(feature["full_feature"], dtype=np.float32)
        if vector.shape != (768,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid train full feature for {key}: {vector.shape}")
        groups[int(label["gt_category_id"])].append(vector)
    return {category: _unit(np.mean(values, axis=0)) for category, values in groups.items()}


def _make_method(name: str, prototypes: dict[int, np.ndarray], *, device: str) -> Any:
    if name == "nearest":
        return NearestPrototype(known_prototypes=prototypes)
    if name == "dpmeans":
        return OnlineDPMeans(known_prototypes=prototypes)
    if name == "phe":
        if not PHE_CHECKPOINT.exists():
            raise FileNotFoundError(PHE_CHECKPOINT)
        return PHETrackAdapter(PHE_CHECKPOINT, radius=2, device=device)
    raise ValueError(name)


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def _load_feature_vector(path: Path) -> np.ndarray:
    row = json.loads(path.read_text(encoding="utf-8"))
    forbidden = {"gt_category_id", "gt_split", "gt_category_name", "gt_match_id"}
    leaked = forbidden & set(row)
    if leaked:
        raise ValueError(f"predicted feature contains evaluator fields: {sorted(leaked)}")
    if str(row.get("source_split")) != "pred":
        raise ValueError(f"unexpected feature source split in {path}: {row.get('source_split')}")
    vector = np.asarray(row["full_feature"], dtype=np.float32)
    if vector.shape != (768,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid predicted full feature {path}: {vector.shape}")
    return vector


def _diagnostic_keys() -> set[str]:
    """Read only opaque public keys from the evaluator diagnostic source.

    This bounded route is intentionally not a formal public replay.  The
    diagnostic source contains no labels, but its membership was obtained by
    evaluator-side geometry, so the output records that selection explicitly.
    """

    if not PRED_DIAGNOSTIC.exists():
        raise FileNotFoundError(PRED_DIAGNOSTIC)
    keys: set[str] = set()
    with PRED_DIAGNOSTIC.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = {"gt_category_id", "gt_split", "gt_category_name", "gt_match_id"}
            if forbidden & set(row):
                raise ValueError("diagnostic predicted source contains evaluator fields")
            key = "pred_" + str(row["sample_id"])
            if key in keys:
                raise ValueError(f"duplicate diagnostic predicted key: {key}")
            keys.add(key)
    return keys


def _causal_pass(method_names: tuple[str, ...], prototypes: dict[int, np.ndarray], device: str, candidate_keys: set[str]) -> tuple[dict[str, Path], dict[str, int]]:
    DECISION_ROOT.mkdir(parents=True, exist_ok=True)
    final_paths = {name: DECISION_ROOT / f"pred_sanity_{name}_full_decisions.jsonl" for name in method_names}
    temporary_paths = {name: _temporary(path) for name, path in final_paths.items()}
    methods = {name: _make_method(name, prototypes, device=device) for name in method_names}
    handles: dict[str, Any] = {}
    counts = {
        "manifest_tracks": 0,
        "diagnostic_candidate_tracks": len(candidate_keys),
        "diagnostic_candidate_seen_in_public_manifest": 0,
        "feature_available_tracks": 0,
        "feature_missing_tracks": 0,
        "public_non_candidate_tracks_skipped": 0,
        "decision_rows": 0,
    }
    previous_order = -1
    try:
        handles = {name: path.open("w", encoding="utf-8") for name, path in temporary_paths.items()}
        for row in _read_jsonl(PUBLIC_MANIFEST):
            counts["manifest_tracks"] += 1
            key = str(row["sample_key"])
            stream_order = int(row["stream_order"])
            if stream_order < previous_order:
                raise ValueError(f"public stream order regressed at {key}: {stream_order} < {previous_order}")
            previous_order = stream_order
            if key not in candidate_keys:
                counts["public_non_candidate_tracks_skipped"] += 1
                continue
            counts["diagnostic_candidate_seen_in_public_manifest"] += 1
            feature_path = FEATURE_ROOT / f"{key}.json"
            done_path = feature_path.with_name(feature_path.name + ".done")
            if not feature_path.exists() or not done_path.exists():
                counts["feature_missing_tracks"] += 1
                continue
            vector = _load_feature_vector(feature_path)
            counts["feature_available_tracks"] += 1
            for name in method_names:
                decision = dict(methods[name].step(vector))
                handles[name].write(json.dumps({
                    "sample_key": key,
                    "stream_order": stream_order,
                    "representation": "full",
                    "decision": decision,
                }, sort_keys=True, separators=(",", ":")) + "\n")
                counts["decision_rows"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    for path in temporary_paths.values():
        if not path.exists():
            raise RuntimeError(f"causal decision temporary output is missing: {path}")
    for name in method_names:
        os.replace(temporary_paths[name], final_paths[name])
    return final_paths, counts


def _ensure_join() -> dict[str, Any]:
    if not JOIN_SCRIPT.exists():
        raise FileNotFoundError(JOIN_SCRIPT)
    result = subprocess.run([sys.executable, str(JOIN_SCRIPT)], cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"evaluator-only predicted join failed with {result.returncode}")
    return json.loads(JOIN_AUDIT.read_text(encoding="utf-8"))


def _load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        key = str(row["sample_key"])
        if key in result:
            raise ValueError(f"duplicate causal decision for {key}: {path}")
        result[key] = dict(row["decision"])
    return result


def _slim_standard(metric: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "old_acc", "new_acc", "h_score", "all_acc", "old_correct", "old_total",
        "new_correct", "new_total", "state_count", "hungarian_mapping",
    )
    return {key: metric[key] for key in keep}


def _metric(rows: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    known_ids = load_ids(ROLE_ROOT / "known_ids.json")
    novel_ids = load_ids(ROLE_ROOT / "unknown_ids_val.json")
    distractor_ids = load_ids(ROLE_ROOT / "distractor_ids.json")
    standard = evaluate_standard(rows, decisions, known_ids=known_ids, novel_ids=novel_ids, distractor_ids=distractor_ids)
    persistent = evaluate_persistent(rows, decisions, known_ids=known_ids, novel_ids=novel_ids, distractor_ids=distractor_ids)
    return {"standard": _slim_standard(standard), "persistent": persistent}


def _predicted_metrics(method_names: tuple[str, ...], decision_paths: dict[str, Path], join_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in method_names:
        decisions = _load_decisions(decision_paths[name])
        rows = [row for row in join_rows if str(row["sample_key"]) in decisions]
        ordered_decisions = [decisions[str(row["sample_key"])] for row in rows]
        metric = _metric(rows, ordered_decisions)
        result[name] = {
            "evaluator_rows": len(rows),
            "matched_join_rows_before_feature_filter": len(join_rows),
            "old_rows": sum(str(row["gt_split"]) == "old" for row in rows),
            "new_rows": sum(str(row["gt_split"]) == "new" for row in rows),
            **metric,
        }
    return result


def _gt_reference(method_names: tuple[str, ...], prototypes: dict[int, np.ndarray], device: str) -> dict[str, Any]:
    """Compute a same-representation GT reference on the four frozen orders."""

    val_labels = {str(row["sample_key"]): row for row in _read_jsonl(VAL_LABELS)}
    order_paths = [
        ("main", MANIFEST_ROOT / "tao_val_gt_tracks.jsonl"),
        ("seed1027", MANIFEST_ROOT / "tao_val_gt_tracks_seed1027.jsonl"),
        ("seed1028", MANIFEST_ROOT / "tao_val_gt_tracks_seed1028.jsonl"),
        ("seed1029", MANIFEST_ROOT / "tao_val_gt_tracks_seed1029.jsonl"),
    ]
    result: dict[str, Any] = {}
    for name in method_names:
        method = _make_method(name, prototypes, device=device)
        orders: dict[str, Any] = {}
        for order_name, path in order_paths:
            if hasattr(method, "reset"):
                method.reset()
            rows: list[dict[str, Any]] = []
            decisions: list[dict[str, Any]] = []
            for row in _read_jsonl(path):
                key = str(row["sample_key"])
                feature = json.loads((TRAIN_FEATURE_ROOT.parent / "val" / f"{key}.json").read_text(encoding="utf-8"))
                vector = np.asarray(feature["full_feature"], dtype=np.float32)
                decisions.append(dict(method.step(vector)))
                label = val_labels[key]
                rows.append(dict(row, gt_category_id=int(label["gt_category_id"]), gt_split=str(label["gt_split"])))
            orders[order_name] = _metric(rows, decisions)
        standard_values = [orders[key]["standard"] for key in orders]
        persistent_values = [orders[key]["persistent"] for key in orders]
        result[name] = {
            "representation": "full",
            "orders": orders,
            "mean": {
                "standard": {
                    key: float(np.mean([value[key] for value in standard_values]))
                    for key in ("old_acc", "new_acc", "h_score", "all_acc")
                },
                "persistent": {
                    key: float(np.mean([value[key] for value in persistent_values]))
                    for key in ("commit_ct", "false_assignment_rate")
                },
            },
        }
    return result


def _table(predicted: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, value in predicted.items():
        rows.append({
            "method": name,
            "representation": "full",
            "evaluator_rows": value["evaluator_rows"],
            "Old": value["standard"]["old_acc"],
            "New": value["standard"]["new_acc"],
            "H": value["standard"]["h_score"],
            "Commit-CT": value["persistent"]["commit_ct"],
            "False Assign": value["persistent"]["false_assignment_rate"],
            "formal": False,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="nearest,dpmeans,phe")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    method_names = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    if not method_names or any(value not in {"nearest", "dpmeans", "phe"} for value in method_names):
        raise SystemExit("--methods must contain nearest, dpmeans, phe")
    if not PUBLIC_MANIFEST.exists():
        raise FileNotFoundError(PUBLIC_MANIFEST)
    out = ensure_output_layout()
    prototypes = _load_prototypes()
    candidate_keys = _diagnostic_keys()
    decision_paths, causal = _causal_pass(method_names, prototypes, args.device, candidate_keys)
    if causal["feature_available_tracks"] == 0:
        atomic_json(OUTPUT, {
            "schema_version": "trackocd.v2.predicted_sanity_benchmark.v1",
            "status": "WAITING_FEATURES",
            "formal_ready": False,
            "representation": "full",
            "causal_pass": causal,
            "test_semantic_accessed": False,
        })
        return 2

    join_audit = _ensure_join()
    join_rows = list(_read_jsonl(JOIN_PATH))
    predicted = _predicted_metrics(method_names, decision_paths, join_rows)
    gt_reference = _gt_reference(method_names, prototypes, args.device)
    feature_coverage = causal["feature_available_tracks"] / max(causal["manifest_tracks"], 1)
    status = "SANITY_COMPLETE" if causal["feature_available_tracks"] == causal["manifest_tracks"] else "SANITY_COMPLETE_PARTIAL_COVERAGE"
    table = _table(predicted)
    result = {
        "schema_version": "trackocd.v2.predicted_sanity_benchmark.v1",
        "status": status,
        "formal_ready": False,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "representation": {
            "selected_for_sanity": "full",
            "encoder": "DINOv2 ViT-B/14",
            "reason": "No legal exported SimOWT/Q0 appearance embedding is available; use one fallback representation only.",
            "used_prefixes": ["full"],
            "p1_p2_p4_p8_p16_simultaneous_benchmark": False,
            "final_physical_frontend_frozen": False,
        },
        "sanity_scope": {
            "status": "EVALUATOR_DIAGNOSTIC_SUBSET_ONLY",
            "candidate_source": str(PRED_DIAGNOSTIC.resolve()),
            "candidate_source_sha256": sha256_file(PRED_DIAGNOSTIC),
            "candidate_membership_was_selected_by_evaluator_side_geometry": True,
            "not_a_full_public_causal_replay": True,
            "reason": "The raw/fragmented stream creates an unbounded anonymous-state scan for these baselines; use this bounded route only to prove evaluator execution and expose the GT-to-predicted sanity gap.",
        },
        "causal_pass": {
            **causal,
            "public_manifest": str(PUBLIC_MANIFEST.resolve()),
            "public_manifest_sha256": sha256_file(PUBLIC_MANIFEST),
            "public_source_sha256": sha256_file(PUBLIC_SOURCE),
            "feature_root": str(FEATURE_ROOT.resolve()),
            "feature_root_is_home_cache_via_symlink": str(FEATURE_ROOT.resolve()).startswith("/home/"),
            "feature_coverage_ratio": feature_coverage,
            "decisions_made_before_evaluator_join": True,
            "decision_files": {name: str(path.resolve()) for name, path in decision_paths.items()},
        },
        "evaluator_subset": {
            "join_audit": str(JOIN_AUDIT.resolve()),
            "join_audit_sha256": sha256_file(JOIN_AUDIT),
            "join_rows": len(join_rows),
            "predicted_rows_with_completed_full_feature": causal["feature_available_tracks"],
            "historical_matched_rows_with_completed_full_feature": len({str(row["sample_key"]) for row in join_rows} & set(_load_decisions(next(iter(decision_paths.values()))))),
            "gt_fields_or_labels_used_by_method": False,
            "historical_matched_stream_used_for_causal_inference": True,
            "formal_public_inference": False,
            "matching_source": join_audit.get("historical_diagnostic_source"),
            "matching_performed_after_public_causal_pass": True,
        },
        "table": table,
        "predicted_metrics": predicted,
        "gt_reference_same_full_representation": gt_reference,
        "gt_to_predicted_gap": {
            "status": "DIAGNOSTIC_ONLY_NOT_COMPARABLE_POPULATIONS",
            "explanation": "Predicted rows are the IoU>=0.5 historical evaluator subset with incomplete-cache filtering; GT reference is all Val on four frozen orders. Differences are a sanity gap, not a controlled frontend comparison.",
            "per_method_predicted_minus_gt_mean": {
                name: {
                    "Old": float(predicted[name]["standard"]["old_acc"] - gt_reference[name]["mean"]["standard"]["old_acc"]),
                    "New": float(predicted[name]["standard"]["new_acc"] - gt_reference[name]["mean"]["standard"]["new_acc"]),
                    "H": float(predicted[name]["standard"]["h_score"] - gt_reference[name]["mean"]["standard"]["h_score"]),
                    "Commit-CT": float(predicted[name]["persistent"]["commit_ct"] - gt_reference[name]["mean"]["persistent"]["commit_ct"]),
                    "False Assign": float(predicted[name]["persistent"]["false_assignment_rate"] - gt_reference[name]["mean"]["persistent"]["false_assignment_rate"]),
                }
                for name in method_names
            },
        },
        "model_input_contract": {
            "input_fields": ["public physical-track stream order", "DINOv2 full track representation", "TRAIN old-only visual prototypes"],
            "gt_category_id": False,
            "gt_split": False,
            "gt_physical_track_id": False,
            "temporal_iou_match": False,
            "future_observations_used": False,
            "text_or_category_logits": False,
        },
        "test_semantic_accessed": False,
    }
    atomic_json(out / "tables/predicted_sanity_benchmark.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
