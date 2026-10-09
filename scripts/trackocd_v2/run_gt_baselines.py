#!/usr/bin/env python3
"""Run fixed vocabulary-free baselines on the frozen GT-track Val streams."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.evaluation.persistent import evaluate_persistent  # noqa: E402
from src.trackocd_v2.evaluation.standard_ocd import evaluate_standard  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, atomic_write_text, ensure_output_layout  # noqa: E402
from src.trackocd_v2.methods.dpmeans import OnlineDPMeans  # noqa: E402
from src.trackocd_v2.methods.nearest_prototype import NearestPrototype  # noqa: E402
from src.trackocd_v2.protocol import load_ids  # noqa: E402


PREFIXES = (1, 2, 4, 8, 16)
MANIFEST_ROOT = OUTPUT_TARGET / "manifests"
FEATURE_ROOT = OUTPUT_TARGET / "features/gt_tracks"
ROLE_ROOT = ROOT / "data/tao_ow_ocd_v1/splits"


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_feature(source_split: str, sample_key: str) -> dict:
    path = FEATURE_ROOT / source_split / (str(sample_key) + ".json")
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else np.zeros_like(value)


def load_prototypes(prefix: int, train_rows: Sequence[dict], train_labels: Dict[str, dict], train_features: Dict[str, dict]) -> Dict[int, np.ndarray]:
    groups: Dict[int, List[np.ndarray]] = defaultdict(list)
    for row in train_rows:
        label = train_labels[str(row["sample_key"])]
        if label["gt_split"] != "old":
            continue
        feature = train_features[str(row["sample_key"])]
        groups[int(label["gt_category_id"])].append(np.asarray(feature["prefix_features"][str(prefix)], dtype=np.float32))
    return {category: unit(np.mean(values, axis=0)) for category, values in groups.items()}


def make_method(name: str, prototypes: Dict[int, np.ndarray]) -> Any:
    if name == "nearest":
        return NearestPrototype(known_prototypes=prototypes)
    if name == "dpmeans":
        return OnlineDPMeans(known_prototypes=prototypes)
    raise ValueError(name)


def run_order(method_name: str, order_name: str, val_rows: Sequence[dict], val_labels: Dict[str, dict], val_features: Dict[str, dict], prototypes: Dict[int, np.ndarray], prefixes: Sequence[int]) -> Tuple[dict, List[dict]]:
    output = {"method": method_name, "order": order_name, "prefixes": {}}
    diagnostics: List[dict] = []
    for prefix in prefixes:
        method = make_method(method_name, prototypes[prefix])
        decisions = []
        evaluator_rows = []
        for row in val_rows:
            key = str(row["sample_key"])
            feature = val_features[key]
            vector = np.asarray(feature["prefix_features"][str(prefix)], dtype=np.float32)
            # Only the anonymous visual vector is passed to the method.  The
            # label sidecar is joined after the causal step for evaluation.
            decision = dict(method.step(vector))
            decisions.append(decision)
            label = val_labels[key]
            evaluator_rows.append(dict(row, gt_category_id=int(label["gt_category_id"]), gt_split=str(label["gt_split"])))
            diagnostics.append({"prefix": int(prefix), "order": order_name, "method": method_name, "sample_key": key, "decision": decision})
        known_ids = load_ids(ROLE_ROOT / "known_ids.json")
        novel_ids = load_ids(ROLE_ROOT / "unknown_ids_val.json")
        distractor_ids = load_ids(ROLE_ROOT / "distractor_ids.json")
        standard = evaluate_standard(evaluator_rows, decisions, known_ids=known_ids, novel_ids=novel_ids, distractor_ids=distractor_ids)
        persistent = evaluate_persistent(evaluator_rows, decisions, known_ids=known_ids, novel_ids=novel_ids, distractor_ids=distractor_ids)
        output["prefixes"][str(prefix)] = {"standard": standard, "persistent": persistent}
    return output, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("nearest", "dpmeans"), required=True)
    parser.add_argument("--prefixes", default="1,2,4,8,16")
    args = parser.parse_args()
    out = ensure_output_layout()
    audit_path = out / "audit/common_feature_audit.json"
    if not audit_path.exists() or json.loads(audit_path.read_text())["prefix_contract"]["status"] != "READY":
        atomic_json(out / ("tables/gt_%s.json" % args.method), {"schema_version": "trackocd.v2.gt_baseline.v1", "status": "WAITING_FEATURES", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        return 2
    prefixes = tuple(int(value) for value in args.prefixes.split(",") if value.strip())
    if any(prefix not in PREFIXES for prefix in prefixes):
        raise SystemExit("prefixes must be a subset of 1,2,4,8,16")
    train_rows = load_jsonl(MANIFEST_ROOT / "tao_train_gt_tracks.jsonl")
    train_labels = {str(row["sample_key"]): row for row in load_jsonl(MANIFEST_ROOT / "private_tao_train_gt_track_labels.jsonl")}
    val_labels = {str(row["sample_key"]): row for row in load_jsonl(MANIFEST_ROOT / "private_tao_val_gt_track_labels.jsonl")}
    train_features = {str(row["sample_key"]): load_feature("train", row["sample_key"]) for row in train_rows}
    val_features = {str(row["sample_key"]): load_feature("val", row["sample_key"]) for row in val_labels.values()}
    prototypes = {prefix: load_prototypes(prefix, train_rows, train_labels, train_features) for prefix in prefixes}
    all_orders = [("main", "tao_val_gt_tracks.jsonl")]
    all_orders.extend(("seed%d" % seed, "tao_val_gt_tracks_seed%d.jsonl" % seed) for seed in (1027, 1028, 1029))
    summaries = []
    all_diagnostics: List[dict] = []
    for order_name, filename in all_orders:
        rows = load_jsonl(MANIFEST_ROOT / filename)
        summary, diagnostics = run_order(args.method, order_name, rows, val_labels, val_features, prototypes, prefixes)
        summaries.append(summary)
        all_diagnostics.extend(diagnostics)
    aggregate = {}
    for prefix in prefixes:
        standard_metrics = [summary["prefixes"][str(prefix)]["standard"] for summary in summaries]
        persistent_metrics = [summary["prefixes"][str(prefix)]["persistent"] for summary in summaries]
        aggregate[str(prefix)] = {
            "standard_mean": {name: float(np.mean([metric[name] for metric in standard_metrics])) for name in ("old_acc", "new_acc", "h_score", "all_acc")},
            "persistent_mean": {name: float(np.mean([metric[name] for metric in persistent_metrics])) for name in ("commit_ct", "false_assignment_rate")},
            "orders": [summary["order"] for summary in summaries],
        }
    result = {
        "schema_version": "trackocd.v2.gt_baseline.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": args.method,
        "prefixes": list(prefixes),
        "fixed_method_parameters": {
            "nearest": {"tau_known": 0.35, "tau_existing": 0.55},
            "dpmeans": {"lambda_distance": 0.45, "tau_known": 0.35},
        },
        "train_known_prototype_categories_by_prefix": {str(prefix): sorted(int(key) for key in prototypes[prefix]) for prefix in prefixes},
        "orders": summaries,
        "aggregate": aggregate,
        "test_semantic_accessed": False,
    }
    atomic_json(out / ("tables/gt_%s.json" % args.method), result)
    diagnostics_path = out / ("diagnostics/gt_%s_decisions.jsonl" % args.method)
    atomic_write_text(diagnostics_path, "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in all_diagnostics))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
