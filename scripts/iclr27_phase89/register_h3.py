#!/usr/bin/env python3
"""Register H3 only from corrected TRAIN-disjoint H2 evidence."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase89"
P88 = ROOT / "outputs/iclr27_phase88"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    selection = P88 / "audit/h2_equal_train_selection_corrected.json"
    frozen = P88 / "audit/h2_equal_frozen_selection_corrected.json"
    if not selection.exists() or not frozen.exists():
        raise RuntimeError("corrected H2 TRAIN selection must be frozen before H3")
    selected = json.loads(selection.read_text())
    if selected.get("status") != "H2_TRAIN_SELECTED":
        raise RuntimeError("H3 requires corrected H2 TRAIN selection")
    rows = selected["h2_equal"]["folds"]
    evidence = []
    for row in rows:
        m = row["metrics"]
        evidence.append({"fold": row["fold"], "selection_score": row["selection_score"],
                         "open_world_false_assignment_rate": m.get("open_world_false_assignment_rate"),
                         "negative_false_merge_rate": m.get("negative_false_merge_rate"),
                         "duplicate_births": m.get("duplicate_births"), "premature_rate": m.get("premature_rate"),
                         "unresolved_rate": m.get("unresolved_rate"),
                         "correct_reuse_recall": m.get("positive_reuse_recall_macro")})
    payload = {
        "schema_version": "trackocd.phase89.h3_preregistration.v1", "phase": 89,
        "route": "H3_HIERARCHICAL_OPEN_WORLD_ROUTER", "status": "REGISTERED_TRAIN_ONLY",
        "hypothesis": "separate KNOWN/OPEN/DEFER/RESET arbitration from OPEN EXISTING/NEW scoring to remove incompatible score-family competition",
        "trigger_source": "corrected H2 TRAIN-disjoint validation only",
        "trigger": {"absolute_open_world_error_materially_high": True, "duplicate_births_materially_high": True, "premature_materially_high": True},
        "evidence": evidence,
        "comparison": {"primary": "C0_REOPT", "c0_reopt": "baseline architecture from same c0v2_fix2_formal 20k checkpoint with fresh AdamW", "h3": "hierarchical router from same checkpoint with fresh AdamW", "updates": 10000, "base_step": 20000, "final_step": 30000, "seed": 88002, "event_tag": "fix2", "support_mode": False},
        "inputs": ["causal track hidden", "quality", "known score summaries", "memory relation summaries", "support summary zero vector", "streak/age/dispersion"],
        "forbidden_inputs": ["category IDs", "text", "semantic IDs", "physical IDs", "future rows/tracks", "held GT", "DEV+", "Q1", "public labels"],
        "loss": {"router_ce": 1.0, "known_ce": 1.0, "open_action_ce": 1.0, "state_relation_bce": 1.0, "open_false_merge": 2.0, "reset_margin": 0.75},
        "decision": "H3 TRAIN selection strictly compares mean exact Phase19R selection_score against C0_REOPT; held replay allowed only if H3 wins",
        "frozen_source_selection": str(frozen.resolve()), "frozen_source_sha256": sha(frozen),
        "public_dev_q1_sealed_accessed": False, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(OUT / "audit/h3_preregistration.json", payload)
    atomic_json(OUT / "audit/overnight_autonomous_state.json", {
        "phase": 89, "status": "REGISTERED_H3", "current_stage": "IMPLEMENT_H3",
        "completed": ["CORRECT_H2_PROTOCOL", "CORRECT_H2_SELECTION", "CORRECT_HELD_EVAL", "CORRECT_STANDARD_METRICS"],
        "pending": ["BUILD_TRUE_HELD_GCD_STREAM", "ASSESS_H2_TRAIN", "TRAIN_H3", "VALIDATE_H3", "SELECT_H3", "H3_HELD", "PHYSICAL_COMPATIBILITY", "FINAL_REPORT"],
        "public_dev_q1_sealed_accessed": False, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__": main()
