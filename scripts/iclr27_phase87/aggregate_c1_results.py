#!/usr/bin/env python3
"""Aggregate C1 support-conditioned diagnostics and freeze the Phase87 decision."""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase87"


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    rows = [load(OUT / "metrics" / f"c1_diag_f{i}.json")["metrics"] for i in range(4)]
    pos = sum(r["positive_events"] for r in rows); neg = sum(r["negative_events"] for r in rows)
    ct = sum(r["commit_ct_correct"] for r in rows)
    fm = sum(round(r["negative_false_merge_rate"] * r["negative_events"]) for r in rows)
    ex = sum(sum(v for a, v in r["action_counts"].items() if a in {"EXISTING", "KNOWN"}) for r in rows)
    cats = set(); vids = set()
    for r in rows:
        cats.update(r["by_category_correct"]); vids.update(r["by_video_correct"])
    aggregate = {"positive_events": pos, "negative_events": neg, "commit_ct_correct": ct, "commit_ct_eligible": pos, "commit_ct_recall": ct / max(1, pos), "existing_precision": ct / max(1, ex), "negative_false_merge_rate": fm / max(1, neg), "category_coverage": len(cats), "video_coverage": len(vids), "folds": rows}
    gate = {"commit_ct_min": ct >= 15, "category_coverage_min": len(cats) >= 5, "video_coverage_min": len(vids) >= 8, "existing_precision_min": aggregate["existing_precision"] >= .70, "negative_false_merge_max": aggregate["negative_false_merge_rate"] <= .15, "known_micro_min": False, "known_macro_min": False}
    gate["all_required"] = all(gate.values())
    result = {"schema_version": "trackocd.phase87.c1_aggregate.v1", "phase": 87, "route": "C1_SUPPORT_INTEGRATION", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "aggregate": aggregate, "formal_gate": gate, "decision": "C1_GATE_FAIL_STOP_PHASE87_CONTROLLER_ROUTES" if not gate["all_required"] else "C1_GATE_PASS_PENDING_COMPATIBILITY", "support_mode": True, "controller_compatibility_run": False, "sealed_or_public_run": False, "public_dev_q1_sealed_accessed": False, "next_action": "FINALIZE_PHASE87_NEGATIVE_CONTROLLER_EVIDENCE" if not gate["all_required"] else "RUN_UNCHANGED_COMPATIBILITY"}
    for target in [OUT / "audit" / "phase87_c1_decision.json", OUT / "metrics" / "c1_aggregate.json"]:
        target.parent.mkdir(parents=True, exist_ok=True); tmp = target.with_name(f".{target.name}.tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, target)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
