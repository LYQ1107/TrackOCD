#!/usr/bin/env python3
"""Register the single TRAIN-only C0/C1 safety-supervision repair."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    # These are the already-frozen TRAIN validation observations from the
    # C0/C1 fair control.  They are evidence for the hypothesis, never held
    # labels and never used for checkpoint selection.
    prereg = {
        "schema_version": "trackocd.phase88.hypothesis.v1",
        "phase": 88,
        "hypothesis_id": "H1_FALSE_MERGE_RESET",
        "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "REGISTERED_TRAIN_ONLY",
        "root_cause_class": "FALSE_MERGE_DOMINANT",
        "secondary_class": "PREMATURE_DOMINANT",
        "evidence": {
            "c0_train_aggregate": {
                "negative_false_merge_rate": 0.699529,
                "premature_rate": 0.643164,
                "unresolved_rate": 0.100215,
                "first_commit_negative_existing_or_known": 0.5518,
                "reset_targets_or_predictions": 0,
            },
            "c1_train_aggregate": {
                "negative_false_merge_rate": 0.500512,
                "premature_rate": 0.629762,
                "unresolved_rate": 0.216148,
                "bridge_use_not_applicable": True,
            },
            "interpretation": "C0 has severe negative false merge and no observed RESET prediction; C1 reduces false merge but trades it for unresolved events and loses the TRAIN selection score. The actionable gap is safety supervision, not a held threshold or memory change.",
            "record_sources": [
                "outputs/iclr27_phase88/validation/c0_continue_fix1_f*_val/records_shard_*.json",
                "outputs/iclr27_phase88/validation/c1_support_fix1_f*_val/records_shard_*.json",
                "outputs/iclr27_phase88/audit/c0_c1_train_selection.json",
            ],
        },
        "exact_change": {
            "code": "src/iclr27_phase88/rollout.py",
            "profile": "h1_false_merge_reset",
            "weights": {
                "action_ce": 1.0,
                "state_relation": 1.0,
                "false_merge_risk": 5.0,
                "new_existing_margin": 1.25,
                "commit_defer_margin": 0.75,
                "reset_margin": 1.25,
                "known_margin": 0.75,
            },
            "unchanged": [
                "event manifest and sampler",
                "support_mode=false (C0 continuation)",
                "StateMemory and transition code",
                "inference thresholds and evaluator",
                "seed/fold/denominator/causal ordering",
            ],
        },
        "training_contract": {
            "base_checkpoints": "outputs/iclr27_phase88/checkpoints/c0v2_fix2_formal_f{0..3}.pt",
            "updates": 30000,
            "start_step": 0,
            "support_mode": False,
            "event_tag": "fix2",
            "seed": 88002,
            "folds": [0, 1, 2, 3],
            "validation": "same TRAIN-disjoint validation and exact Phase19R selection_score",
            "devices": "bounded workers on available GPUs only",
        },
        "expected_behavior": [
            "reduce negative false merge and premature commits on TRAIN validation",
            "retain or improve existing F1/selection score without all-DEFER collapse",
            "produce nonzero RESET/safety response under TRAIN reset augmentation",
        ],
        "failure_criterion": [
            "TRAIN selection mean not above frozen C0 continuation",
            "false merge remains dominant without safety improvement",
            "all-raw/all-defer or nonfinite/policy collapse",
        ],
        "forbidden": [
            "DEV+/Q1/public/sealed labels",
            "held 76+76 diagnostic for selection",
            "threshold, StateMemory, transition, denominator or evaluator changes",
            "category/text/physical-ID/semantic-ID/future inputs",
        ],
        "source_code_sha256_before_route": {
            "rollout.py": sha(ROOT / "src/iclr27_phase88/rollout.py"),
            "train_controller.py": sha(ROOT / "scripts/iclr27_phase88/train_controller.py"),
        },
    }
    out = OUT / "audit/hypothesis_1_preregistration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(out)


if __name__ == "__main__":
    main()
