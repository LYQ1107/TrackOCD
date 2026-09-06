#!/usr/bin/env python3
"""Write the fixed comparison after the one registered C0 repair."""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase87"


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def aggregate(prefix: str) -> dict:
    rows = [load(OUT / "metrics" / f"{prefix}_diag_f{i}.json")["metrics"] for i in range(4)]
    pos = sum(r["positive_events"] for r in rows); neg = sum(r["negative_events"] for r in rows)
    ct = sum(r["commit_ct_correct"] for r in rows)
    fm = sum(round(r["negative_false_merge_rate"] * r["negative_events"]) for r in rows)
    ex = sum(sum(v for a, v in r["action_counts"].items() if a in {"EXISTING", "KNOWN"}) for r in rows)
    cats = set(); vids = set()
    for r in rows:
        cats.update(r["by_category_correct"]); vids.update(r["by_video_correct"])
    return {"positive_events": pos, "negative_events": neg, "commit_ct_correct": ct, "commit_ct_eligible": pos, "commit_ct_recall": ct / max(1, pos), "existing_precision": ct / max(1, ex), "negative_false_merge_rate": fm / max(1, neg), "category_coverage": len(cats), "video_coverage": len(vids), "folds": rows}


def main() -> None:
    c0 = load(OUT / "metrics" / "c0_aggregate.json")["diagnostic_76_plus_76"]
    repair = aggregate("c0_repair1")
    gate = {"commit_ct_min": repair["commit_ct_correct"] >= 15, "category_coverage_min": repair["category_coverage"] >= 5, "video_coverage_min": repair["video_coverage"] >= 8, "existing_precision_min": repair["existing_precision"] >= .70, "negative_false_merge_max": repair["negative_false_merge_rate"] <= .15}
    gate["all_required"] = all(gate.values())
    result = {"schema_version": "trackocd.phase87.repair1_aggregate.v1", "phase": 87, "route": "C0_FALSE_MERGE_REPAIR1", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "c0": c0, "repair1": repair, "comparison": {"commit_ct_delta": repair["commit_ct_correct"] - c0["commit_ct_correct"], "existing_precision_delta": repair["existing_precision"] - c0["existing_precision"], "negative_false_merge_delta": repair["negative_false_merge_rate"] - c0["negative_false_merge_rate"]}, "formal_gate": gate, "decision": "REPAIR1_GATE_FAIL_STOP_C0_ROUTE" if not gate["all_required"] else "REPAIR1_GATE_PASS", "next_action": "DO_NOT_RUN_C1_OR_CONTROLLER_ON_HELD" if not gate["all_required"] else "REGISTER_C1_SUPPORT_ONLY", "public_dev_q1_sealed_accessed": False, "held_used_for_checkpoint_selection": False}
    for target in [OUT / "audit" / "phase87_repair1_decision.json", OUT / "metrics" / "repair1_aggregate.json"]:
        target.parent.mkdir(parents=True, exist_ok=True); tmp = target.with_name(f".{target.name}.tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, target)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
