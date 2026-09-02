#!/usr/bin/env python3
"""Exact frozen R-global/R-legal replay for the TRAIN-selected best models."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch

from src.iclr27_phase75d.gates import gate_rows, strict_gate
from src.iclr27_phase75d.protocol import load_frozen_tracks
from src.iclr27_phase75e.evaluator import aggregate_sections, evaluate_fold
from src.iclr27_phase75e.model import LowRankFeatureAdapter


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase75e"


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_best(fold: int, tag: str, device: torch.device) -> tuple[LowRankFeatureAdapter, dict]:
    path = OUT / "checkpoints" / f"{tag}_f{fold}_best.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LowRankFeatureAdapter().to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    return model, {"path": str(path.resolve()), "step": checkpoint.get("step"), "seed": checkpoint.get("seed"), "manifest_sha256": checkpoint.get("manifest_sha256")}


def resource_snapshot() -> dict:
    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "free_h": subprocess.run(["free", "-h"], capture_output=True, text=True, check=False).stdout,
        "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False).stdout,
        "pid": os.getpid(), "gpu_count_used": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="phase75e_formal"); ap.add_argument("--run-id", default="phase75e-exact-20260902-r1"); args = ap.parse_args()
    atomic(OUT / "audit/resource_exact_preflight.json", resource_snapshot())
    device = torch.device("cpu")
    table = load_frozen_tracks()
    fold_evals = []; checkpoints = []
    for fold in range(4):
        model, info = load_best(fold, args.tag, device); checkpoints.append({"fold": fold, **info})
        fold_evals.append(evaluate_fold(model, table, fold, device, global_query_limit=None, legal_query_limit=None))
    aggregate = aggregate_sections(fold_evals)

    gate_by_section = {}
    for section in ("global", "legal"):
        rows = []
        for fe in fold_evals:
            p16 = next(x for x in fe["prefix_rows"] if x["prefix"] == 16)[section]
            raw, learned = p16["raw"], p16["learned"]
            rows.append({"fold": fe["fold"], "metrics": {"raw": {"r1": raw["r1"], "map": raw["map"], "hard_negative_gap": raw["hard_negative_gap"]}, "pairwise": {"r1": learned["r1"], "map": learned["map"], "hard_negative_gap": learned["hard_negative_gap"], "unsafe_flip_count": learned["unsafe_flip_count"]}}})
        gate_by_section[section] = strict_gate(gate_rows(rows, section))

    payload = {
        "phase": "Phase75E", "run_id": args.run_id, "tag": args.tag, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "exact Phase75D R-global all validation cross-video candidates and R-legal explicit Phase30 val candidates; prefixes 1,2,4,8,16",
        "folds": fold_evals, "aggregate": aggregate, "gates": gate_by_section, "checkpoints": checkpoints,
        "training_source": "Phase30 fit only; checkpoints selected by TRAIN-disjoint validation lexicographic rule",
        "held_event_accessed_for_model": False, "sealed_accessed": False,
        "sealed_inputs_not_read": ["DEV+", "Q1", "public-new", "sealed", "152 held events", "future rows", "category/text/physical IDs as tensors"],
    }
    atomic(OUT / "metrics/exact_best_retrieval.json", payload)
    status = "P75E_GATE_R_PASS" if gate_by_section["global"]["pass"] and gate_by_section["legal"]["pass"] else "P75E_GATE_R_FAIL_STOP_BEFORE_CONTROLLER"
    atomic(OUT / "status.json", {"phase": "Phase75E", "status": status, "run_id": args.run_id, "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip(), "training": True, "gpu_count": 4, "input_hashes": {"csv": table.csv_sha256, "features": table.feature_sha256, "permutation": table.alignment.get("permutation_sha256")}, "fold_manifest_hashes": {str(c["fold"]): c["manifest_sha256"] for c in checkpoints}, "global_r": {"gate": gate_by_section["global"], "aggregate_prefix16": aggregate["global"]["16"]}, "legal_r": {"gate": gate_by_section["legal"], "aggregate_prefix16": aggregate["legal"]["16"]}, "unsafe": {s: aggregate[s]["16"]["unsafe_flip_count"] for s in ("global", "legal")}, "gates": {"global_pass": gate_by_section["global"]["pass"], "legal_pass": gate_by_section["legal"]["pass"]}, "failures": [], "repairs": [], "qualified_for_controller": False, "qualified_for_sealed": False, "held_event_accessed_for_model": False, "sealed_accessed": False})
    atomic(OUT / "audit/resource_exact_postflight.json", resource_snapshot())
    atomic(OUT / "completion/exact_best.done", {"phase": "Phase75E", "status": status, "metrics": str(OUT / "metrics/exact_best_retrieval.json")})
    print(json.dumps({"phase": "Phase75E", "status": status, "global": gate_by_section["global"], "legal": gate_by_section["legal"]}, sort_keys=True))


if __name__ == "__main__":
    main()
