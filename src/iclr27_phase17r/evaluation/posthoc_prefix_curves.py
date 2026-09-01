"""Add descriptive causal-prefix curves to the frozen Phase17R audit.

This script consumes only the already-written public audit decisions.  It does
not run a model, change a threshold/denominator, or participate in selection.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs/iclr27_phase17r"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def as_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_curve(rows: list[dict[str, str]], eligible_keys: set[str]) -> list[dict[str, Any]]:
    births: dict[int, dict[str, int]] = {}
    annotations = []
    for position, row in enumerate(rows):
        semantic_id = int(row["semantic_id"])
        if row["action"] == "new":
            births[semantic_id] = {
                "category": int(row["gt_category_id_common"]) if row["gt_role_common"] == "novel" else -1,
                "video": int(row["video_id"]),
                "position": position,
            }
        birth = births.get(semantic_id)
        existing_correct = bool(
            row["action"] == "existing"
            and row["gt_role_common"] == "novel"
            and birth
            and birth["category"] == int(row["gt_category_id_common"])
        )
        ct_correct = bool(
            row["row_key"] in eligible_keys
            and existing_correct
            and birth
            and birth["video"] != int(row["video_id"])
        )
        annotations.append(
            {
                "row": row,
                "prefix_count": int(row["causal_prefix_count"]),
                "target_observable": as_bool(row["assigned"]) and float(row["row_iou"]) >= 0.5,
                "predicted_observable": as_bool(row["predicted_observable"]),
                "known_correct": row["action"] == "known"
                and int(row["semantic_id"]) == int(row["gt_category_id_common"]),
                "existing_correct": existing_correct,
                "ct_correct": ct_correct,
            }
        )

    curve = []
    for minimum in sorted({x["prefix_count"] for x in annotations}):
        selected = [x for x in annotations if x["prefix_count"] >= minimum]
        known = [x for x in selected if x["row"]["gt_role_common"] == "supported_known"]
        novel = [x for x in selected if x["row"]["gt_role_common"] == "novel"]
        fp = [x for x in selected if x["row"]["gt_role_common"] == "fp"]
        existing = [x for x in selected if x["row"]["action"] == "existing"]
        ct = [x for x in selected if x["row"]["row_key"] in eligible_keys]
        predicted_obs = [x for x in selected if x["predicted_observable"]]
        target_obs = [x for x in selected if x["target_observable"]]
        obs_tp = sum(x["target_observable"] for x in predicted_obs)
        curve.append(
            {
                "minimum_causal_prefix_count": minimum,
                "rows": len(selected),
                "known_rows": len(known),
                "known_accuracy": safe_ratio(sum(x["known_correct"] for x in known), len(known)),
                "novel_rows": len(novel),
                "novel_existing_action_rate": safe_ratio(
                    sum(x["row"]["action"] == "existing" for x in novel), len(novel)
                ),
                "predicted_existing": len(existing),
                "predicted_existing_precision": safe_ratio(
                    sum(x["existing_correct"] for x in existing), len(existing)
                ),
                "fp_rows": len(fp),
                "fp_known_false_accept_rate": safe_ratio(
                    sum(x["row"]["action"] == "known" for x in fp), len(fp)
                ),
                "target_observable_rows": len(target_obs),
                "predicted_observable_rows": len(predicted_obs),
                "observability_precision": safe_ratio(obs_tp, len(predicted_obs)),
                "observability_recall": safe_ratio(obs_tp, len(target_obs)),
                "fixed_ct_eligible": len(ct),
                "fixed_ct_correct": sum(x["ct_correct"] for x in ct),
                "fixed_ct_recall": safe_ratio(sum(x["ct_correct"] for x in ct), len(ct)),
            }
        )
    return curve


def main() -> None:
    audit_path = OUT / "eval/public_final_audit.json"
    audit = json.loads(audit_path.read_text())
    denominators = json.loads((OUT / "eval/fixed_ct_denominators.json").read_text())
    sources = {}
    for order in audit["orders"]:
        seed = int(order["seed"])
        csv_path = OUT / f"csv/public_final_audit_decisions_{seed}.csv"
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == audit["complete_rows"]
        ranks = [int(row["event_rank"]) for row in rows]
        assert all(a < b for a, b in zip(ranks, ranks[1:]))
        eligible = set(denominators["denominators"]["audit"][str(seed)]["row_keys"])
        curve = build_curve(rows, eligible)
        assert curve[0]["known_accuracy"] == order["metrics"]["known_occurrence_accuracy"]
        assert curve[0]["predicted_existing"] == order["metrics"]["predicted_existing"]
        assert curve[0]["fixed_ct_eligible"] == order["metrics"]["fixed_ct"]["eligible"]
        assert curve[0]["fixed_ct_correct"] == order["metrics"]["fixed_ct"]["correct"]
        order["metrics"]["causal_prefix_curve"] = curve
        sources[str(seed)] = {
            "decision_csv": str(csv_path.resolve()),
            "decision_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "points": len(curve),
        }

    audit["causal_prefix_curve_contract"] = {
        "definition": "Cumulative audit subset with causal_prefix_count >= minimum; count is current causal observation count (age + 1).",
        "computed_from_frozen_decisions_after_full_result": True,
        "used_for_training_calibration_or_selection": False,
        "thresholds_changed": False,
        "fixed_ct_denominators_changed": False,
        "sources": sources,
    }
    atomic_json(audit_path, audit)
    print(json.dumps({"orders": len(audit["orders"]), "points": [len(x["metrics"]["causal_prefix_curve"]) for x in audit["orders"]]}, indent=2))


if __name__ == "__main__":
    main()
