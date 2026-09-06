#!/usr/bin/env python3
"""Aggregate frozen C0 replay and write its preregistered decision."""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase87"


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def atomic(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    val = [load(OUT / "metrics" / f"c0_val_f{i}.json") for i in range(4)]
    diag = [load(OUT / "metrics" / f"c0_diag_repair1_f{i}.json") for i in range(4)]
    dm = [x["metrics"] for x in diag]
    positive = sum(x["positive_events"] for x in dm)
    negative = sum(x["negative_events"] for x in dm)
    correct = sum(x["commit_ct_correct"] for x in dm)
    false_merge = sum(round(x["negative_false_merge_rate"] * x["negative_events"]) for x in dm)
    false_commit = sum(round(x["negative_false_commit_rate"] * x["negative_events"]) for x in dm)
    existing_predictions = sum(sum(v for a, v in x["action_counts"].items() if a in {"EXISTING", "KNOWN"}) for x in dm)
    category_set = set(); video_set = set()
    for x in dm:
        category_set.update(x["by_category_correct"])
        video_set.update(x["by_video_correct"])
    train_improvement = [x["metrics"]["commit_ct_recall"] > 0.0 for x in val]
    aggregate = {
        "positive_events": positive,
        "negative_events": negative,
        "commit_ct_correct": correct,
        "commit_ct_eligible": positive,
        "commit_ct_recall": correct / max(1, positive),
        "category_coverage": len(category_set),
        "video_coverage": len(video_set),
        "existing_precision": correct / max(1, existing_predictions),
        "negative_false_merge_rate": false_merge / max(1, negative),
        "negative_false_commit_rate": false_commit / max(1, negative),
        "folds": dm,
    }
    gate = {
        "commit_ct_min": aggregate["commit_ct_correct"] >= 15,
        "category_coverage_min": aggregate["category_coverage"] >= 5,
        "video_coverage_min": aggregate["video_coverage"] >= 8,
        "existing_precision_min": aggregate["existing_precision"] >= 0.70,
        "negative_false_merge_max": aggregate["negative_false_merge_rate"] <= 0.15,
        "known_micro_min": False,
        "known_macro_min": False,
        "all_required": False,
    }
    gate["all_required"] = all(gate.values())
    result = {"schema_version": "trackocd.phase87.c0_aggregate.v1", "phase": 87, "route": "C0_CAUSAL_PERSISTENT_CONTROLLER", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "train_validation": [{"fold": x["fold"], "metrics": x["metrics"]} for x in val], "diagnostic_76_plus_76": aggregate, "formal_gate": gate, "decision": "C0_TRAIN_IMPROVEMENT_BUT_FORMAL_OCD_GATE_FAIL" if not gate["all_required"] else "C0_GATE_PASS", "train_validation_improved_over_all_defer": train_improvement, "public_dev_q1_sealed_accessed": False, "held_used_for_checkpoint_selection": False, "next_action": "REGISTER_FALSE_MERGE_WEIGHT_REPAIR" if not gate["all_required"] else "RUN_C1_SUPPORT_CONDITIONED_ROUTE"}
    atomic(OUT / "audit" / "phase87_c0_decision.json", result)
    atomic(OUT / "metrics" / "c0_aggregate.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
