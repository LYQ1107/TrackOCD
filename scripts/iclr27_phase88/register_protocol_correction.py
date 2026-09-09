#!/usr/bin/env python3
"""Register Phase88R protocol corrections without mutating historical artifacts."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    historical = {
        "h2_equal_train_selection": OUT / "audit/h2_equal_train_selection.json",
        "h2_equal_frozen_selection": OUT / "audit/h2_equal_frozen_selection.json",
        "h2_equal_held_support_on": OUT / "diagnostic/h2_equal_76plus76_diagnostic/metrics.json",
        "standard_stream_v1": OUT / "audit/standard_gcd_stream_metrics.json",
    }
    for path in historical.values():
        if not path.exists():
            raise FileNotFoundError(path)
    c0 = []
    h2 = []
    for fold in range(4):
        for route in ("c0_continue_fix1", "h2_equal"):
            metrics = OUT / "metrics" / f"{route}_f{fold}.json"
            d = json.loads(metrics.read_text())
            if int(d.get("start_step", -1)) != 20000 or int(d.get("updates", -1)) != 30000:
                raise RuntimeError(f"EQUAL_BUDGET_ENDPOINT_MISMATCH {metrics}")
            if bool(d.get("support_mode", False)):
                raise RuntimeError(f"TRAIN_SUPPORT_MODE_MISMATCH {metrics}")
        c0.append(json.loads((OUT / "metrics" / f"c0_continue_fix1_f{fold}.json").read_text()))
        h2.append(json.loads((OUT / "metrics" / f"h2_equal_f{fold}.json").read_text()))
    payload = {
        "phase": 88,
        "source_head": "58ced135cf75b058774403b32fc6e8ef54a0deed",
        "status": "CORRECTION_IN_PROGRESS",
        "issues": [
            "H2 selection compared against C0 20k instead of frozen C0_CONTINUE 30k",
            "held H2 diagnostic forced support_mode=true although H2_EQUAL protocol is false",
            "standard stream evaluator forced support_mode=true",
            "standard stream aggregate performed cross-fold Hungarian despite fold-local tokens",
            "current pseudo-new stream is TRAIN pseudo-novel rather than true held novel",
            "fold0 H2 continuation lacks exact sampler/rollout RNG continuation",
        ],
        "old_h2_held_status": "INVALID_FOR_H2_ATTRIBUTION_SUPPORT_MODE_MISMATCH",
        "old_standard_stream_status": "AUXILIARY_INVALID_FOR_FINAL_STANDARD_METRIC",
        "historical_artifacts_preserved": {k: {"path": str(v.resolve()), "sha256": sha(v)} for k, v in historical.items()},
        "matched_control_audit": {
            "status": "F0_MATCHED_CONTROL_ALREADY_EXISTS",
            "route": "c0_continue_fix1",
            "all_folds": True,
            "same_base_step": 20000,
            "same_final_step": 30000,
            "same_seed_family": "88002+fold",
            "same_event_tag": "fix2",
            "same_base_checkpoint_family": "c0v2_fix2_formal_f{fold}.pt",
            "fold0_exact_rng_continuation": False,
            "fold0_deterministic_missing_state_policy": True,
            "retraining_required": False,
            "metrics": [{"fold": f, "checkpoint_sha256": c0[f]["checkpoint_sha256"], "resumed_from": c0[f]["resumed_from"], "sampler_state_persisted": c0[f].get("sampler_state_persisted"), "rollout_rng_state_persisted": c0[f].get("rollout_rng_state_persisted")} for f in range(4)],
        },
        "endpoint_audit": {
            "c0_continue_fix1": [{"fold": f, "start_step": c0[f]["start_step"], "updates": c0[f]["updates"]} for f in range(4)],
            "h2_equal": [{"fold": f, "start_step": h2[f]["start_step"], "updates": h2[f]["updates"]} for f in range(4)],
        },
        "public_dev_q1_sealed_accessed": False,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(OUT / "audit/phase88r_protocol_correction_v2.json", payload)
    atomic_json(OUT / "audit/overnight_autonomous_state.json", {
        "phase": 88,
        "status": "CORRECT_H2_PROTOCOL",
        "completed": [],
        "current_stage": "CORRECT_H2_PROTOCOL",
        "next_stage": "CORRECT_H2_SELECTION",
        "public_dev_q1_sealed_accessed": False,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
