#!/usr/bin/env python3
"""Write the Phase76R frozen Phase75 contract errata artifacts."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path

from src.iclr27_phase76r.errata import correct_teacher_authorization

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76r"


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def main() -> None:
    dglobal = json.loads((ROOT / "outputs/iclr27_phase75d/metrics/global_r.json").read_text())
    dlegal = json.loads((ROOT / "outputs/iclr27_phase75d/metrics/legal_support_r.json").read_text())
    e_status = json.loads((ROOT / "outputs/iclr27_phase75e/status.json").read_text())
    e_cfg = json.loads((ROOT / "configs/iclr27_phase75e/phase75e_rank8.json").read_text())
    global_folds = [
        {"fold": r["fold"], "delta_r1": r["delta_r1"], "delta_map": r["delta_map"], "delta_hard_gap": r["delta_hard_gap"]}
        for r in dglobal["gate"]["rows"]
    ]
    legal_folds = [
        {"fold": r["fold"], "delta_r1": r["delta_r1"], "delta_map": r["delta_map"], "delta_hard_gap": r["delta_hard_gap"]}
        for r in dlegal["gate"]["rows"]
    ]
    err = correct_teacher_authorization(global_folds, legal_folds)
    atomic(OUT / "audit/teacher_authorization_erratum.json", {"phase": "Phase76R", **err})
    # Phase75E used Adam(lr=4e-5) directly; no scheduler was instantiated.
    atomic(OUT / "audit/optimizer_contract_erratum.json", {
        "phase": "Phase76R", "phase75e_actual_optimizer": "torch.optim.Adam", "phase75e_actual_lr": 4e-5,
        "phase75e_actual_scheduler": None, "registered_warmup_steps": e_cfg.get("warmup_steps"),
        "registered_warmup_executed": False, "statement": "actual Phase75E LR schedule = constant 4e-5; registered/reported warmup = not executed",
    })
    seeds = {str(f): 750500 + f for f in range(4)}
    atomic(OUT / "audit/seed_contract_erratum.json", {
        "phase": "Phase76R", "phase75e_reported_seed": 42, "phase75e_worker_seed_formula": "750500 + fold", "phase75e_actual_fold_seeds": seeds,
    })
    selected = []
    for fold in range(4):
        p = ROOT / "outputs/iclr27_phase75e/metrics" / f"phase75e_formal_f{fold}.json"
        d = json.loads(p.read_text())
        selected.append({"fold": fold, "best_step": d["best_step"], "selection_key": next(v["selection_key"] for v in d["validation_history"] if v["step"] == d["best_step"])})
    atomic(OUT / "audit/selection_contract_erratum.json", {
        "phase": "Phase76R", "phase75e_selection_key": ["legal_unsafe", "-legal_map", "-legal_hard_gap", "-global_map"],
        "omitted_hard_constraints": ["global_unsafe", "global_r1", "global_hard_gap", "raw_adapt_cosine"],
        "interpretation": "selected checkpoints were legal-first, not Pareto-safe global/legal checkpoints", "selected": selected,
    })
    atomic(OUT / "audit/status.json", {
        "phase": "Phase76R", "status": "CONTRACT_ERRATA_RECORDED_READY_FOR_PARETO_AUDIT", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase75d_teacher_old": err["old_result"], "phase75d_teacher_corrected": err["corrected_result"],
        "phase75e_status": e_status["status"], "training": False, "gpu_count": 0,
        "held_event_accessed_for_model": False, "sealed_accessed": False,
        "next_action": "exact p16 audit of all 120 Phase75E checkpoints",
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip(),
    })
    print(json.dumps({"phase": "Phase76R", "status": "CONTRACT_ERRATA_RECORDED_READY_FOR_PARETO_AUDIT", "teacher": err}, sort_keys=True))


if __name__ == "__main__":
    main()

