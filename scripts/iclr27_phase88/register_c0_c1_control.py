#!/usr/bin/env python3
"""Register the only permitted C0/C1 fair continuation comparison."""
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
    selection_path = OUT / "audit/c0v2_train_selection.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "FROZEN_TRAIN_ONLY":
        raise RuntimeError("C0/C1 registration requires frozen TRAIN selection")
    folds = []
    for row in selection["folds"]:
        folds.append({
            "fold": int(row["fold"]),
            "init_checkpoint": row["checkpoint"],
            "init_checkpoint_sha256": row["checkpoint_sha256"],
            "base_step": int(row["step"]),
            "seed": 88002 + int(row["fold"]),
            "c0_tag": f"c0_continue_f{row['fold']}",
            "c1_tag": f"c1_support_f{row['fold']}",
        })
    manifest_path = OUT / "manifests/causal_event_v2_fix2_manifest.json"
    contract = {
        "schema_version": "trackocd.phase88.c0_c1_control_registration.v1",
        "phase": 88,
        "status": "REGISTERED_TRAIN_ONLY_FAIR_CONTROL",
        "hypothesis": "With identical frozen C0v2 initialization, seed, sampler, optimizer and 10000 additional updates, real causal 8-D support evidence (C1) improves validation selection metrics over zero-support continuation (C0), without changing the state/action contract.",
        "control": {
            "C0_CONTINUE": {"support_mode": False, "support_features": "exact_zero_8d"},
            "C1_SUPPORT": {"support_mode": True, "support_features": "computed_causal_8d"},
        },
        "equal_exposure": {
            "additional_updates": 10000,
            "max_updates_absolute": 30000,
            "optimizer": "checkpoint_restored_AdamW",
            "sampler": "checkpoint_restored_balanced_causal_event_sampler",
            "seed_base": 88002,
            "event_tag": "fix2",
            "checkpoint_interval": 2000,
            "video_category_disjoint": True,
        },
        "folds": folds,
        "selection": "TRAIN_disjoint_validation only; never held/DEV+/Q1/public",
        "failure_rules": [
            "nonfinite loss or invalid checkpoint",
            "protocol/causal/support input contract change",
            "resource floor breach handled only by task-owned stop and resume",
            "no held metric may select a branch",
        ],
        "input_manifest_sha256": sha(manifest_path),
        "selection_sha256": sha(selection_path),
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_or_text_as_model_input": False,
        "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(OUT / "audit/c0_continue_vs_c1_support_preregistration.json", contract)
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
