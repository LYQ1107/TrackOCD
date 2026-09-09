#!/usr/bin/env python3
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main() -> None:
    d = {
        "schema_version": "trackocd.phase88.hypothesis.v1",
        "phase": 88,
        "hypothesis_id": "H2_KNOWN_BRANCH_SUPPRESSION",
        "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "REGISTERED_TRAIN_ONLY",
        "root_cause_class": "FALSE_MERGE_DOMINANT_KNOWN_BRANCH",
        "evidence": {
            "h1_train_aggregate": {
                "negative_false_merge_rate": 0.7448194197750148,
                "premature_rate": 0.6308466548253404,
                "unresolved_rate": 0.11160449970396684,
                "negative_first_commit_existing_or_known": 0.7448194197750148,
            },
            "code_audit": "The existing false_merge_risk term penalizes state_logits only; known_margin is not populated for non-KNOWN targets. H1 increased state/reset penalties but did not suppress unrelated KNOWN logits, and negative false-merge remained dominant.",
            "record_source": "outputs/iclr27_phase88/validation/h1_trainval_f*_val/records_shard_*.json",
        },
        "exact_change": {
            "code": "src/iclr27_phase88/rollout.py",
            "profile": "h2_known_suppression",
            "new_term": "For every TRAIN non-KNOWN target, compare the strongest legal known logit against the desired EXISTING/NEW/DEFER/RESET logit with softplus margin.",
            "weight": 1.5,
            "base_weights_unchanged": {
                "action_ce": 1.0,
                "state_relation": 1.0,
                "false_merge_risk": 2.0,
                "new_existing_margin": 0.75,
                "commit_defer_margin": 0.75,
                "reset_margin": 0.75,
                "known_margin": 0.75,
            },
            "unchanged": [
                "event manifest/sampler and 10 percent TRAIN reset augmentation",
                "StateMemory/transitions, action semantics and inference thresholds",
                "support_mode=false, seed/fold/denominator/evaluator",
            ],
        },
        "training_contract": {
            "base_checkpoints": "outputs/iclr27_phase88/checkpoints/c0v2_fix2_formal_f{0..3}.pt",
            "updates": 30000,
            "seed": 88002,
            "event_tag": "fix2",
            "support_mode": False,
            "folds": [0, 1, 2, 3],
            "validation": "same four TRAIN-disjoint val manifests and frozen Phase19R selection_score",
        },
        "expected_behavior": [
            "lower negative false merge caused by KNOWN predictions",
            "retain positive existing F1 and avoid all-DEFER/unresolved collapse",
            "show nonzero known-suppression loss only on legal TRAIN non-KNOWN steps",
        ],
        "failure_criterion": [
            "mean TRAIN selection_score not above frozen C0",
            "negative false merge remains dominant without coverage/precision gain",
            "known branch collapses on positive/known TRAIN stream or all-DEFER/all-NEW policy collapse",
        ],
        "forbidden": [
            "held 76+76/public/DEV+/Q1/sealed labels",
            "threshold, StateMemory, transition, denominator or evaluator changes",
            "category/text/physical-ID/semantic-ID/future model inputs",
        ],
        "source_code_sha256": {
            "rollout.py": sha(ROOT / "src/iclr27_phase88/rollout.py"),
            "train_controller.py": sha(ROOT / "scripts/iclr27_phase88/train_controller.py"),
        },
    }
    out = OUT / "audit/hypothesis_2_preregistration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(out)

if __name__ == "__main__":
    main()
