#!/usr/bin/env python3
"""Write the preregistered Phase90 protocol before any new training."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase90"


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
    source_head = "04013a26ea5f99d98474f3201304a17aa02208dd"
    phase89_report = ROOT / "docs/iclr27_phase89/TRACKOCD_OVERNIGHT_AUTONOMOUS_REPORT.md"
    base = ROOT / "outputs/iclr27_phase88/checkpoints"
    folds = []
    for fold in range(4):
        ckpt = base / f"c0v2_fix2_formal_f{fold}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        folds.append({"fold": fold, "checkpoint": str(ckpt.resolve()), "sha256": sha(ckpt), "step": 20000})
    payload = {
        "schema_version": "trackocd.phase90.preregistration.v1",
        "phase": 90,
        "source_head": source_head,
        "goal": "protocol-correct H3 reproduction and calibrated safe persistent OCD",
        "max_gpus": 4,
        "gpu_pool": [1, 3, 7, 8],
        "support_mode": False,
        "base_step": 20000,
        "final_step": 30000,
        "additional_updates": 10000,
        "seed": 88002,
        "event_tag": "fix2",
        "held_used_for_selection": False,
        "public_dev_q1_sealed_accessed": False,
        "hypotheses": {
            "P90-A": "H3 positive signal survives exact optimizer-resume semantics.",
            "P90-B": "branch-normalized probabilistic hierarchy reduces score-family calibration error.",
            "P90-C": "if premature remains high, causal evidence maturity improves commit timing.",
        },
        "routes": {
            "H3_REPRO": {"architecture": "phase89_h3_unchanged", "loss": "phase89_h3_router_unchanged"},
            "C0_REOPT_REPRO": {"architecture": "phase88_baseline", "loss": "phase89_c0_baseline_unchanged"},
        },
        "optimizer_contract": {
            "initial_base_resume": "fresh AdamW and --reinit-optimizer exactly once",
            "resource_resume": "restore optimizer/sampler/RNG; no --reinit-optimizer",
            "required_state": ["optimizer", "sampler_state", "rollout_rng_state", "python_rng_state", "numpy_rng_state", "torch_rng_state"],
        },
        "selection": {
            "primary": "mean exact Phase19R TRAIN-disjoint selection_score",
            "calibrated_secondary": "mean score > H3_REPRO, mean open-world false assignment < H3_REPRO, reuse >= 0.95 H3_REPRO",
            "maturity_trigger": "mean TRAIN premature_rate >= 0.40",
        },
        "gates": {
            "final_commit_ct": 15,
            "category_coverage": 5,
            "video_coverage": 8,
            "existing_precision": 0.70,
            "open_world_false_assignment": 0.15,
            "known_micro": 0.206,
            "known_macro": 0.139,
        },
        "base_checkpoints": folds,
        "source_phase89_report": str(phase89_report.resolve()),
        "source_phase89_report_sha256": sha(phase89_report),
        "forbidden": ["DEV+", "Q1", "public new-model labels", "sealed labels", "future rows/tracks", "category/text/semantic/physical ID model inputs", "tracking retraining", "threshold sweep"],
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(OUT / "audit/phase90_preregistration.json", payload)
    atomic_json(OUT / "audit/phase90_autonomous_state.json", {"phase": 90, "state": "REGISTER", "status": "REGISTERED", "updated_utc": payload["generated_utc"], "public_dev_q1_sealed_accessed": False})
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
