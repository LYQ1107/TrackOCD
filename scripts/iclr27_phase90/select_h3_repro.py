#!/usr/bin/env python3
"""Train-only selection of the protocol-correct H3 reproduction."""
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
        for b in iter(lambda: f.read(1 << 20), b): h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")
    os.replace(tmp, path)


def row(route: str, fold: int) -> dict:
    metrics_path = OUT / "validation" / f"{route}_f{fold}_val/final_metrics.json"
    d = json.loads(metrics_path.read_text()); m = d["metrics"]
    ckpt = OUT / "checkpoints" / f"{route}_f{fold}.pt"
    train_metrics = json.loads((OUT / "metrics" / f"{route}_f{fold}.json").read_text())
    keys = ("selection_score", "existing_precision", "existing_recall", "existing_f1", "new_precision", "new_recall", "new_f1", "open_world_false_assignment_rate", "anonymous_false_merge_rate", "known_capture_error_rate", "premature_rate", "unresolved_rate", "duplicate_births", "positive_reuse_recall_macro", "known_micro", "known_macro")
    return {"fold": fold, "route": route, "metrics": {k: m.get(k) for k in keys}, "checkpoint": str(ckpt.resolve()), "checkpoint_sha256": sha(ckpt), "validation": str(metrics_path.resolve()), "validation_sha256": sha(metrics_path), "selection_score": float(m["selection_score"]), "train_metrics": {"optimizer_policy": train_metrics.get("optimizer_policy"), "optimizer_restored": train_metrics.get("optimizer_restored"), "resume_state_complete": train_metrics.get("resume_state_complete")}}


def main() -> None:
    c0 = [row("c0_reopt_repro", f) for f in range(4)]
    h3 = [row("h3_repro", f) for f in range(4)]
    c0_mean = sum(x["selection_score"] for x in c0) / 4.0; h3_mean = sum(x["selection_score"] for x in h3) / 4.0
    payload = {"schema_version": "trackocd.phase90.repro_selection.v1", "phase": 90, "status": "H3_REPRO_TRAIN_SELECTED" if h3_mean > c0_mean else "C0_REOPT_REPRO_TRAIN_SELECTED", "candidate": "H3_REPRO" if h3_mean > c0_mean else "C0_REOPT_REPRO", "selection_rule": {"primary": "mean exact Phase19R TRAIN-disjoint selection_score", "strict_h3_win": "h3_mean > c0_mean", "held_or_public_used": False}, "c0_reopt_repro": {"folds": c0, "mean_selection_score": c0_mean}, "h3_repro": {"folds": h3, "mean_selection_score": h3_mean}, "per_fold_score_delta_h3_minus_c0": [h3[i]["selection_score"] - c0[i]["selection_score"] for i in range(4)], "h3_folds_won": sum(h3[i]["selection_score"] > c0[i]["selection_score"] for i in range(4)), "endpoint": {"base_step": 20000, "final_step": 30000, "additional_updates": 10000, "support_mode": False}, "public_dev_q1_sealed_accessed": False, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    atomic_json(OUT / "audit/h3_repro_train_selection.json", payload)
    if payload["candidate"] == "H3_REPRO":
        atomic_json(OUT / "audit/h3_repro_frozen_selection.json", {"schema_version": "trackocd.phase90.frozen_repro.v1", "phase": 90, "status": "FROZEN_TRAIN_ONLY", "candidate": "H3_REPRO", "selection_source": str((OUT / "audit/h3_repro_train_selection.json").resolve()), "selection_sha256": sha(OUT / "audit/h3_repro_train_selection.json"), "folds": h3, "support_mode": False, "public_dev_q1_sealed_accessed": False, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    atomic_json(OUT / "audit/phase90_autonomous_state.json", {"phase": 90, "state": "SELECT_H3_REPRO", "status": payload["status"], "candidate": payload["candidate"], "completed": ["REGISTER", "FIX_OPTIMIZER_RESUME", "H3_REPRO_TRAIN", "C0_REOPT_REPRO_TRAIN", "H3_REPRO_VALIDATE", "C0_REOPT_REPRO_VALIDATE", "SELECT_H3_REPRO"], "pending": ["IMPLEMENT_CALIBRATED", "TRAIN_CALIBRATED", "TRAIN_SELECTION", "MATURITY_IF_TRIGGERED", "FINAL_HELD", "FINAL_REPORT"], "public_dev_q1_sealed_accessed": False, "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    print(json.dumps({"status": payload["status"], "candidate": payload["candidate"], "c0_mean": c0_mean, "h3_mean": h3_mean, "fold_deltas": payload["per_fold_score_delta_h3_minus_c0"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
