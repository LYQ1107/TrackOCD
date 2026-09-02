#!/usr/bin/env python3
"""Exact validation for Phase76G best routers, preserving raw fallback."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase76g.router import GroupRobustRelationRouter
from src.iclr27_phase76s.evaluator import evaluate_examples, p16

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76g"


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_model(path: Path) -> GroupRobustRelationRouter:
    try: payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: payload = torch.load(path, map_location="cpu")
    model = GroupRobustRelationRouter().cpu(); model.load_state_dict(payload["model"]); model.eval(); return model


def fold_eval(fold: int, tag: str) -> dict[str, Any]:
    examples = ROOT / "outputs/iclr27_phase76s/examples" / f"examples_f{fold}.json"
    checkpoint = OUT / "checkpoints" / f"{tag}_f{fold}_best.pt"
    if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
    payload = json.loads(examples.read_text()); model = load_model(checkpoint); result = evaluate_examples(model, payload["val"], torch.device("cpu"))
    row = {"fold": fold, "tag": tag, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "examples_sha256": hashlib.sha256(examples.read_bytes()).hexdigest(), "p16": p16(result), "result": result}
    atomic(OUT / "metrics" / f"exact_{tag}_f{fold}.json", {"phase": "Phase76G", "fold_result": row, "sealed_accessed": False, "public_or_dev_accessed": False}); return row


def aggregate(tag: str) -> dict[str, Any]:
    rows = [json.loads((OUT / "metrics" / f"exact_{tag}_f{fold}.json").read_text())["fold_result"] for fold in range(4)]
    keys = ("r1", "map", "hard_negative_gap", "raw_r1", "raw_map", "raw_hard_negative_gap", "delta_r1", "delta_map", "delta_hard_gap", "teacher_agreement", "teacher_use_rate", "router_help_rate")
    agg = {key: float(np.mean([float(row["p16"][key]) for row in rows])) for key in keys}; agg.update({"unsafe_flip_count": int(sum(int(row["p16"]["unsafe_flip_count"]) for row in rows)), "queries": int(sum(int(row["p16"]["queries"]) for row in rows)), "worst_fold_delta_r1": float(min(float(row["p16"]["delta_r1"]) for row in rows)), "worst_fold_delta_map": float(min(float(row["p16"]["delta_map"]) for row in rows))})
    gates = {"delta_r1_ge_0.02": agg["delta_r1"] >= 0.02, "delta_map_ge_0.01": agg["delta_map"] >= 0.01, "unsafe_zero": agg["unsafe_flip_count"] == 0, "hard_gap_non_worse_each_fold": all(float(row["p16"]["delta_hard_gap"]) >= -1e-7 for row in rows), "r1_three_of_four": sum(float(row["p16"]["delta_r1"]) >= 0.02 for row in rows) >= 3, "map_three_of_four": sum(float(row["p16"]["delta_map"]) >= 0.01 for row in rows) >= 3, "worst_fold_delta_ge_minus_0.01": agg["worst_fold_delta_r1"] >= -0.01 and agg["worst_fold_delta_map"] >= -0.01}
    decision = "PHASE76G_GATE_R_PASS" if all(gates.values()) else "PHASE76G_GATE_R_FAIL_ROUTE_TO_PHASE76X"
    obj = {"phase": "Phase76G", "tag": tag, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "decision": decision, "fold_results": rows, "aggregate_p16": agg, "gates": gates, "controller_run": False, "state_memory_run": False, "sealed_accessed": False, "public_or_dev_accessed": False, "protocol": "Phase76G rotating fit-category meta-holdout; raw-preserving HELP-only route"}
    atomic(OUT / "metrics" / f"phase76g_{tag}_exact_retrieval.json", obj); atomic(OUT / "audit" / "phase76g_decision.json", obj); atomic(OUT / "completion" / f"exact_{tag}.done", {"phase": "Phase76G", "decision": decision, "metrics": str(OUT / "metrics" / f"phase76g_{tag}_exact_retrieval.json")}); return obj


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=None); ap.add_argument("--tag", default="g1_formal"); ap.add_argument("--aggregate", action="store_true"); args = ap.parse_args()
    if args.aggregate: print(json.dumps(aggregate(args.tag), sort_keys=True)); return
    folds = range(4) if args.fold is None else [args.fold]; print(json.dumps({"phase": "Phase76G", "tag": args.tag, "folds": [fold_eval(int(fold), args.tag)["fold"] for fold in folds], "status": "fold_saved"}, sort_keys=True))


if __name__ == "__main__": main()
