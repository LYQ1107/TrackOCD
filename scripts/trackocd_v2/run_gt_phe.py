#!/usr/bin/env python3
"""Run the existing PHE-Track checkpoint on v2 GT-track Val streams."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.evaluation.persistent import evaluate_persistent  # noqa: E402
from src.trackocd_v2.evaluation.standard_ocd import evaluate_standard  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, atomic_write_text, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.methods.phe_track import PHETrackAdapter  # noqa: E402
from src.trackocd_v2.protocol import load_ids  # noqa: E402


PREFIXES = (1, 2, 4, 8, 16)
MANIFEST_ROOT = OUTPUT_TARGET / "manifests"
FEATURE_ROOT = OUTPUT_TARGET / "features/gt_tracks"
ROLE_ROOT = ROOT / "data/tao_ow_ocd_v1/splits"
CHECKPOINT = ROOT / "runs/phe_track/dinov2_seed1027/checkpoint.pth"


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_features(split: str, keys: List[str]) -> Dict[str, dict]:
    return {
        key: json.loads((FEATURE_ROOT / split / (key + ".json")).read_text(encoding="utf-8"))
        for key in keys
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    out = ensure_output_layout()
    table_path = out / "tables/gt_phe.json"
    audit_path = out / "audit/common_feature_audit.json"
    if not audit_path.exists() or json.loads(audit_path.read_text())["prefix_contract"]["status"] != "READY":
        atomic_json(table_path, {"schema_version": "trackocd.v2.gt_phe.v1", "status": "WAITING_FEATURES", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        return 2
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    val_labels = {str(row["sample_key"]): row for row in load_jsonl(MANIFEST_ROOT / "private_tao_val_gt_track_labels.jsonl")}
    all_val_keys = sorted(val_labels)
    features = load_features("val", all_val_keys)
    known_ids = load_ids(ROLE_ROOT / "known_ids.json")
    novel_ids = load_ids(ROLE_ROOT / "unknown_ids_val.json")
    distractor_ids = load_ids(ROLE_ROOT / "distractor_ids.json")
    adapters = {prefix: PHETrackAdapter(CHECKPOINT, radius=args.radius, device=args.device) for prefix in PREFIXES}
    checkpoint_class_ids = sorted(adapters[PREFIXES[0]].class_ids)
    orders = [("main", "tao_val_gt_tracks.jsonl")]
    orders.extend(("seed%d" % seed, "tao_val_gt_tracks_seed%d.jsonl" % seed) for seed in (1027, 1028, 1029))
    by_order = []
    diagnostics: List[dict] = []
    for order_name, filename in orders:
        rows = load_jsonl(MANIFEST_ROOT / filename)
        evaluator_rows = [dict(row, gt_category_id=int(val_labels[str(row["sample_key"])] ["gt_category_id"]), gt_split=str(val_labels[str(row["sample_key"])] ["gt_split"])) for row in rows]
        prefix_results = {}
        for prefix in PREFIXES:
            adapter = adapters[prefix]
            adapter.reset()
            decisions = []
            for row in evaluator_rows:
                key = str(row["sample_key"])
                decision = adapter.step(np.asarray(features[key]["prefix_features"][str(prefix)], dtype=np.float32))
                decisions.append(decision)
                diagnostics.append({"method": "phe", "order": order_name, "prefix": prefix, "sample_key": key, "decision": decision})
            prefix_results[str(prefix)] = {
                "standard": evaluate_standard(evaluator_rows, decisions, known_ids=known_ids, novel_ids=novel_ids, distractor_ids=distractor_ids),
                "persistent": evaluate_persistent(evaluator_rows, decisions, known_ids=known_ids, novel_ids=novel_ids, distractor_ids=distractor_ids),
            }
        by_order.append({"order": order_name, "prefixes": prefix_results})
    aggregate = {}
    for prefix in PREFIXES:
        standard = [row["prefixes"][str(prefix)]["standard"] for row in by_order]
        persistent = [row["prefixes"][str(prefix)]["persistent"] for row in by_order]
        aggregate[str(prefix)] = {
            "standard_mean": {name: float(np.mean([row[name] for row in standard])) for name in ("old_acc", "new_acc", "h_score", "all_acc")},
            "persistent_mean": {name: float(np.mean([row[name] for row in persistent])) for name in ("commit_ct", "false_assignment_rate")},
        }
    result = {
        "schema_version": "trackocd.v2.gt_phe.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "radius": args.radius,
        "device": args.device,
        "known_checkpoint_class_count": len(checkpoint_class_ids),
        "checkpoint_class_ids": checkpoint_class_ids,
        "known_role_count": len(known_ids),
        "checkpoint_known_coverage": float(len(set(checkpoint_class_ids) & known_ids) / max(len(known_ids), 1)),
        "orders": by_order,
        "aggregate": aggregate,
        "test_semantic_accessed": False,
    }
    atomic_json(table_path, result)
    atomic_write_text(out / "diagnostics/gt_phe_decisions.jsonl", "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in diagnostics))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
