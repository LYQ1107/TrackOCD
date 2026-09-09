#!/usr/bin/env python3
"""Freeze H2_EQUAL only from same-fold TRAIN-disjoint validation."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import argparse
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


def row(fold: int, route: str) -> dict:
    """Read one explicit equal-budget endpoint; never infer C0 from H2."""
    checkpoint_names = {
        "c0v2_fix2_formal": f"c0v2_fix2_formal_f{fold}.pt",
        "c0_continue_fix1": f"c0_continue_fix1_f{fold}.pt",
        "h2_equal": f"h2_equal_f{fold}.pt",
    }
    if route not in checkpoint_names:
        raise ValueError(f"unknown route: {route}")
    tag = f"{route}_f{fold}_val"
    p = OUT / "validation" / tag / "final_metrics.json"
    d = json.loads(p.read_text()); m = d["metrics"]
    metrics_path = OUT / "metrics" / f"{route}_f{fold}.json"
    training = json.loads(metrics_path.read_text())
    if int(training.get("start_step", -1)) != 20000 or int(training.get("updates", -1)) != 30000:
        raise RuntimeError(f"EQUAL_BUDGET_ENDPOINT_MISMATCH {metrics_path}")
    if bool(training.get("support_mode", False)):
        raise RuntimeError(f"TRAIN_SUPPORT_MODE_MISMATCH {metrics_path}")
    checkpoint = OUT / "checkpoints" / checkpoint_names[route]
    return {"fold": fold, "route": route, "tag": tag, "metrics": m,
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint),
            "training_metrics": str(metrics_path.resolve()), "training_metrics_sha256": sha(metrics_path),
            "start_step": int(training["start_step"]), "updates": int(training["updates"]),
            "support_mode": bool(training.get("support_mode", False)),
            "resume_state_repair_required": bool(training.get("resume_state_repair_required", False)),
            "selection_score": float(m["selection_score"]),
            "validation": str(p.resolve()), "validation_sha256": sha(p),
            "anonymous_false_merge_rate": m.get("anonymous_false_merge_rate"),
            "known_capture_error_rate": m.get("known_capture_error_rate"),
            "open_world_false_assignment_rate": m.get("open_world_false_assignment_rate")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c0-route", choices=["c0v2_fix2_formal", "c0_continue_fix1"], default="c0v2_fix2_formal")
    ap.add_argument("--output", default=str(OUT / "audit/h2_equal_train_selection.json"))
    ap.add_argument("--frozen-output", default=str(OUT / "audit/h2_equal_frozen_selection.json"))
    args = ap.parse_args()
    c0 = [row(f, args.c0_route) for f in range(4)]
    h2 = [row(f, "h2_equal") for f in range(4)]
    c0_scores = [float(x["metrics"]["selection_score"]) for x in c0]
    h2_scores = [float(x["metrics"]["selection_score"]) for x in h2]
    c0_mean = sum(c0_scores) / 4.0; h2_mean = sum(h2_scores) / 4.0
    payload = {
        "schema_version": "trackocd.phase88.h2_equal_train_selection.v2",
        "phase": 88, "route": "H2_EQUAL_BUDGET",
        "status": "H2_TRAIN_SELECTED" if h2_mean > c0_mean else "C0_TRAIN_SELECTED",
        "comparison": {"c0_route": args.c0_route, "h2_route": "h2_equal", "equal_budget": True},
        "selection_rule": {"primary": "mean exact Phase19R selection_score across same-fold TRAIN-disjoint validation", "strict_h2_win": "h2_mean > c0_mean", "held_or_public_used": False},
        "c0": {"folds": c0, "mean_selection_score": c0_mean},
        "h2_equal": {"folds": h2, "mean_selection_score": h2_mean},
        "per_fold_score_delta_h2_minus_c0": [h2_scores[i] - c0_scores[i] for i in range(4)],
        "h2_folds_won": sum(h2_scores[i] > c0_scores[i] for i in range(4)),
        "h2_equal_budget": {"base_step": 20000, "final_step": 30000, "additional_updates": 10000, "loss_profile": "h2_known_suppression", "support_mode": False},
        "legacy_f0_resume_state_repair": "f0 base lacked sampler/rollout state; equal route metrics flag deterministic sampler repair; no claim of exact state continuation for f0",
        "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_or_text_as_model_input": False,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    output_path = Path(args.output)
    frozen_path = Path(args.frozen_output)
    atomic_json(output_path, payload)
    if payload["status"] == "H2_TRAIN_SELECTED":
        frozen = {
            "schema_version": "trackocd.phase88.frozen_h2_equal_selection.v2",
            "phase": 88, "status": "FROZEN_TRAIN_ONLY", "candidate": "H2_EQUAL_BUDGET",
            "selection_source": str(output_path.resolve()),
            "selection_sha256": sha(output_path),
            "folds": h2, "selected_fold_checkpoints": h2,
            "registered_formal_endpoint": {"base_step": 20000, "final_step": 30000, "additional_updates": 10000},
            "support_mode": False,
            "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_or_text_as_model_input": False,
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        atomic_json(frozen_path, frozen)
    print(json.dumps({k: payload[k] for k in ("status", "c0", "h2_equal", "per_fold_score_delta_h2_minus_c0", "h2_folds_won")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
