#!/usr/bin/env python3
"""Run and select the v2 probability-hierarchy controller on GT Val tracks."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.evaluation.persistent import evaluate_persistent  # noqa: E402
from src.trackocd_v2.evaluation.standard_ocd import evaluate_standard  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.methods.safe_controller import SafePersistentController  # noqa: E402
from src.trackocd_v2.protocol import load_ids  # noqa: E402


PREFIXES = (1, 2, 4, 8, 16)
MANIFEST_ROOT = OUTPUT_TARGET / "manifests"
FEATURE_ROOT = OUTPUT_TARGET / "features/gt_tracks"
ROLE_ROOT = ROOT / "data/tao_ow_ocd_v1/splits"
OUTPUT = OUTPUT_TARGET / "tables/gt_safe_controller.json"
SELECTION = OUTPUT_TARGET / "audit/safe_controller_val_selection.json"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _features(split: str, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep only the causal aggregates needed by this controller.

    The per-track JSON also contains all frame embeddings.  Loading those
    arrays for every candidate would needlessly consume several GiB, so the
    compact view is built once and discards frame-level payloads immediately.
    """

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["sample_key"])
        payload = json.loads((FEATURE_ROOT / split / f"{key}.json").read_text(encoding="utf-8"))
        result[key] = {
            "prefix_features": {
                str(prefix): np.asarray(payload["prefix_features"][str(prefix)], dtype=np.float32)
                for prefix in PREFIXES
            },
            "prefix_observations_used": {
                str(prefix): int(payload.get("prefix_observations_used", {}).get(str(prefix), min(prefix, len(row["frame_ids"]))))
                for prefix in PREFIXES
            },
            "quality_mean": float(np.mean(payload.get("quality", [1.0]))),
        }
    return result


def _prototypes(train_rows: list[dict[str, Any]], train_labels: dict[str, dict[str, Any]], train_features: dict[str, dict[str, Any]], prefix: int) -> dict[int, np.ndarray]:
    groups: dict[int, list[np.ndarray]] = {}
    for row in train_rows:
        key = str(row["sample_key"])
        label = train_labels[key]
        if str(label["gt_split"]) != "old":
            continue
        groups.setdefault(int(label["gt_category_id"]), []).append(np.asarray(train_features[key]["prefix_features"][str(prefix)], dtype=np.float32))
    result = {}
    for category, values in groups.items():
        vector = np.mean(values, axis=0)
        result[category] = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return result


def _run_candidate(candidate: dict[str, Any], *, prefixes: tuple[int, ...], train_rows: list[dict[str, Any]], train_labels: dict[str, dict[str, Any]], train_features: dict[str, dict[str, Any]], val_features: dict[str, dict[str, Any]], val_labels: dict[str, dict[str, Any]], orders: dict[str, list[dict[str, Any]]], known: set[int], novel: set[int], distractor: set[int]) -> dict[str, Any]:
    order_results = []
    for order_name, rows in orders.items():
        prefix_results = {}
        for prefix in prefixes:
            controller = SafePersistentController(known_prototypes=_prototypes(train_rows, train_labels, train_features, prefix), **candidate)
            evaluator_rows = []
            decisions = []
            for row in rows:
                key = str(row["sample_key"])
                feature = val_features[key]
                used = int(feature["prefix_observations_used"][str(prefix)])
                quality = float(feature["quality_mean"])
                decision = controller.step(np.asarray(feature["prefix_features"][str(prefix)], dtype=np.float32), observations=used, quality=quality)
                decisions.append(decision)
                label = val_labels[key]
                evaluator_rows.append(dict(row, gt_category_id=int(label["gt_category_id"]), gt_split=str(label["gt_split"])))
            prefix_results[str(prefix)] = {
                "standard": evaluate_standard(evaluator_rows, decisions, known_ids=known, novel_ids=novel, distractor_ids=distractor),
                "persistent": evaluate_persistent(evaluator_rows, decisions, known_ids=known, novel_ids=novel, distractor_ids=distractor),
            }
        order_results.append({"order": order_name, "prefixes": prefix_results})
    aggregate = {}
    for prefix in prefixes:
        standards = [item["prefixes"][str(prefix)]["standard"] for item in order_results]
        persistents = [item["prefixes"][str(prefix)]["persistent"] for item in order_results]
        aggregate[str(prefix)] = {
            "standard_mean": {key: float(np.mean([item[key] for item in standards])) for key in ("old_acc", "new_acc", "h_score", "all_acc")},
            "persistent_mean": {key: float(np.mean([item[key] for item in persistents])) for key in ("commit_ct", "false_assignment_rate")},
        }
    return {"parameters": candidate, "orders": order_results, "aggregate": aggregate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", default="1,2,4,8,16")
    args = parser.parse_args()
    out = ensure_output_layout()
    prefixes = tuple(int(value) for value in args.prefixes.split(",") if value.strip())
    train_rows = _jsonl(MANIFEST_ROOT / "tao_train_gt_tracks.jsonl")
    val_rows = _jsonl(MANIFEST_ROOT / "tao_val_gt_tracks.jsonl")
    train_labels = {str(row["sample_key"]): row for row in _jsonl(MANIFEST_ROOT / "private_tao_train_gt_track_labels.jsonl")}
    val_labels = {str(row["sample_key"]): row for row in _jsonl(MANIFEST_ROOT / "private_tao_val_gt_track_labels.jsonl")}
    train_features = _features("train", train_rows)
    val_features = _features("val", val_rows)
    orders = {"main": val_rows}
    for seed in (1027, 1028, 1029):
        orders[f"seed{seed}"] = _jsonl(MANIFEST_ROOT / f"tao_val_gt_tracks_seed{seed}.jsonl")
    known = load_ids(ROLE_ROOT / "known_ids.json")
    novel = load_ids(ROLE_ROOT / "unknown_ids_val.json")
    distractor = load_ids(ROLE_ROOT / "distractor_ids.json")

    candidates = []
    for tau_known in (0.55, 0.70, 0.85):
        for tau_existing in (0.55, 0.70, 0.85):
            for min_observations in (1, 2, 4):
                candidates.append({
                    "temperature": 0.08,
                    "maturity_temperature": 1.0,
                    "tau_known": tau_known,
                    "tau_existing": tau_existing,
                    "tau_new": 0.45,
                    "min_observations": min_observations,
                })
    baseline = json.loads((out / "tables/gt_nearest.json").read_text(encoding="utf-8"))
    baseline_p16 = baseline["aggregate"]["16"]
    evaluated = []
    for candidate in candidates:
        result = _run_candidate(candidate, prefixes=(16,), train_rows=train_rows, train_labels=train_labels, train_features=train_features, val_features=val_features, val_labels=val_labels, orders=orders, known=known, novel=novel, distractor=distractor)
        metrics = result["aggregate"]["16"]
        h = metrics["standard_mean"]["h_score"]
        ct = metrics["persistent_mean"]["commit_ct"]
        far = metrics["persistent_mean"]["false_assignment_rate"]
        evaluated.append({"candidate": candidate, "result": result, "selection_metrics": {"h_score": h, "commit_ct": ct, "false_assignment_rate": far}, "safe_against_nearest": ct > baseline_p16["persistent_mean"]["commit_ct"] and far <= baseline_p16["persistent_mean"]["false_assignment_rate"] and h >= baseline_p16["standard_mean"]["h_score"] - 0.01})
    safe = [item for item in evaluated if item["safe_against_nearest"]]
    if safe:
        chosen = max(safe, key=lambda item: (item["selection_metrics"]["commit_ct"], item["selection_metrics"]["h_score"], -item["selection_metrics"]["false_assignment_rate"]))
        selection_status = "SAFE_CONTROLLER_SELECTED"
    else:
        chosen = max(evaluated, key=lambda item: (item["selection_metrics"]["h_score"], -item["selection_metrics"]["false_assignment_rate"]))
        selection_status = "NO_SAFE_IMPROVEMENT_INCUMBENT_NEAREST"
    chosen_full = _run_candidate(chosen["candidate"], prefixes=prefixes, train_rows=train_rows, train_labels=train_labels, train_features=train_features, val_features=val_features, val_labels=val_labels, orders=orders, known=known, novel=novel, distractor=distractor)
    selection = {
        "schema_version": "trackocd.v2.safe_controller_selection.v1",
        "status": selection_status,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "chosen_parameters": chosen["candidate"],
        "chosen_p16_metrics": chosen["selection_metrics"],
        "incumbent_nearest_p16": baseline_p16,
        "selection_rule": "Val p16: Commit-CT strictly above Nearest, False Assignment no worse, H-score within 0.01; otherwise retain Nearest as incumbent",
        "candidate_count": len(evaluated),
        "test_semantic_accessed": False,
    }
    atomic_json(SELECTION, selection)
    result = {
        "schema_version": "trackocd.v2.safe_controller_benchmark.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "SafePersistentController",
        "prefixes": list(prefixes),
        "selection": str(SELECTION.resolve()),
        "selection_sha256": sha256_file(SELECTION),
        "chosen_result": chosen_full,
        "all_p16_candidates": [{"parameters": item["candidate"], "selection_metrics": item["selection_metrics"], "safe_against_nearest": item["safe_against_nearest"]} for item in evaluated],
        "test_semantic_accessed": False,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
