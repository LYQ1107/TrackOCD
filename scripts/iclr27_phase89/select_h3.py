#!/usr/bin/env python3
"""Select H3 only from matched TRAIN-disjoint C0_REOPT comparison."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase89"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")
    os.replace(tmp, path)


def row(route: str, fold: int) -> dict:
    p = OUT / "validation" / f"{route}_f{fold}_val/final_metrics.json"
    d = json.loads(p.read_text()); m = d["metrics"]
    ckpt = OUT / "checkpoints" / f"{route}_f{fold}.pt"
    return {"fold": fold, "route": route, "metrics": m, "checkpoint": str(ckpt.resolve()),
            "checkpoint_sha256": sha(ckpt), "validation": str(p.resolve()), "validation_sha256": sha(p),
            "selection_score": float(m["selection_score"]),
            "architecture": json.loads((OUT / "metrics" / f"{route}_f{fold}.json").read_text()).get("architecture")}


def main() -> None:
    c0 = [row("c0_reopt", f) for f in range(4)]
    h3 = [row("h3_router", f) for f in range(4)]
    c0_mean = sum(x["selection_score"] for x in c0) / 4.0
    h3_mean = sum(x["selection_score"] for x in h3) / 4.0
    payload = {
        "schema_version": "trackocd.phase89.h3_selection.v1", "phase": 89,
        "status": "H3_TRAIN_SELECTED" if h3_mean > c0_mean else "C0_REOPT_TRAIN_SELECTED",
        "candidate": "H3_ROUTER" if h3_mean > c0_mean else "C0_REOPT",
        "selection_rule": {"primary": "mean exact Phase19R TRAIN-disjoint selection_score", "strict_h3_win": "h3_mean > c0_mean", "held_or_public_used": False},
        "c0_reopt": {"folds": c0, "mean_selection_score": c0_mean},
        "h3_router": {"folds": h3, "mean_selection_score": h3_mean},
        "per_fold_score_delta_h3_minus_c0": [h3[i]["selection_score"] - c0[i]["selection_score"] for i in range(4)],
        "h3_folds_won": sum(h3[i]["selection_score"] > c0[i]["selection_score"] for i in range(4)),
        "endpoint": {"base_step": 20000, "final_step": 30000, "additional_updates": 10000, "support_mode": False},
        "public_dev_q1_sealed_accessed": False, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(OUT / "audit/h3_train_selection.json", payload)
    if payload["candidate"] == "H3_ROUTER":
        frozen = {"schema_version": "trackocd.phase89.frozen_h3_selection.v1", "phase": 89,
                  "status": "FROZEN_TRAIN_ONLY", "candidate": "H3_ROUTER",
                  "selection_source": str((OUT / "audit/h3_train_selection.json").resolve()),
                  "selection_sha256": sha(OUT / "audit/h3_train_selection.json"),
                  "folds": h3, "selected_fold_checkpoints": h3, "support_mode": False,
                  "public_dev_q1_sealed_accessed": False, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(OUT / "audit/h3_frozen_selection.json", frozen)
    atomic_json(OUT / "audit/overnight_autonomous_state.json", {
        "phase": 89, "status": payload["status"], "current_stage": "SELECT_H3",
        "candidate": payload["candidate"], "completed": ["TRAIN_H3", "TRAIN_C0_REOPT", "VALIDATE_H3", "VALIDATE_C0_REOPT", "SELECT_H3"],
        "pending": ["H3_HELD", "PHYSICAL_COMPATIBILITY", "FINAL_REPORT"] if payload["candidate"] == "H3_ROUTER" else ["ONE_H3_TRAIN_EVIDENCE_REPAIR", "FINAL_REPORT"],
        "public_dev_q1_sealed_accessed": False, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    print(json.dumps({"status": payload["status"], "candidate": payload["candidate"], "c0_mean": c0_mean, "h3_mean": h3_mean, "fold_deltas": payload["per_fold_score_delta_h3_minus_c0"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
