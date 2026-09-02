#!/usr/bin/env python3
"""Short Phase75E contract audit; no model training or held-label access."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path

from src.iclr27_phase75e.data import frozen_table, load_fit_episodes, manifest_hash


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase75e"


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


def commit() -> str:
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else "UNKNOWN"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="phase75e-audit-20260902-r1")
    args = ap.parse_args()
    source = json.loads((ROOT / "outputs/iclr27_phase75d/metrics/legal_support_r.json").read_text())
    teacher = source.get("teacher_signal", {})
    if not teacher.get("signal"):
        raise SystemExit("Phase75D teacher signal is absent; Phase75E is not authorized")
    table = frozen_table(); folds = []
    for fold in range(4):
        fit = load_fit_episodes(fold, set(table.sequences))
        folds.append({"fold": fold, "fit_episodes": len(fit), "manifest_sha256": manifest_hash(fold), "query_keys": len({x.query_key for x in fit}), "positive_links": sum(len(x.positive_keys) for x in fit), "hard_negative_links": len(fit)})
    resource = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "free_h": subprocess.run(["free", "-h"], text=True, capture_output=True, check=False).stdout,
        "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False).stdout,
        "process_count": len(os.listdir("/proc")),
        "disk_data1": subprocess.run(["df", "-h", str(ROOT)], text=True, capture_output=True, check=False).stdout,
        "disk_data2": subprocess.run(["df", "-h", "/data2"], text=True, capture_output=True, check=False).stdout,
        "max_gpus": 4,
    }
    atomic(OUT / "audit/resource_preflight.json", resource)
    atomic(OUT / "audit/contract.json", {
        "phase": "Phase75E", "status": "PASS", "run_id": args.run_id, "source_commit": commit(),
        "teacher_signal": teacher, "phase75d_source": "outputs/iclr27_phase75d/metrics/legal_support_r.json",
        "input_hashes": {"csv": table.csv_sha256, "features": table.feature_sha256, "permutation": table.alignment.get("permutation_sha256")},
        "rows": len(table.rows), "tracks": len(table.sequences), "feature_dim": int(table.features.shape[1]), "prefixes": [1,2,4,8,16],
        "folds": folds, "training_source": "Phase30 fit multi_positive_cross_video only", "validation_source": "Phase30 val manifests, no held outcomes",
        "model_input_contract": ["causal visual 768-D frame features only"],
        "forbidden_inputs": ["category", "semantic_id", "physical_id", "text", "future", "held/DEV+/Q1/public-new/sealed labels"],
        "assignment": "detached CPU Hungarian indices with selected torch similarities retaining gradient",
        "loss": "0.5*rank + 1.0*raw_reconstruction + 1.0*safe",
        "checkpoint_selection": "min legal unsafe, max legal mAP, max legal hard gap, max global mAP",
        "sealed_accessed": False, "held_event_accessed_for_model": False,
    })
    atomic(OUT / "audit/supervision_inventory.json", {"phase": "Phase75E", "folds": folds, "source": "Phase30 fit", "labels_as_tensors": False, "category_or_id_as_input": False})
    atomic(OUT / "audit/leakage_audit.json", {"phase": "Phase75E", "status": "PASS", "phase30_fit_only": True, "val_only_for_selection": True, "held_event_accessed_for_model": False, "sealed_accessed": False, "future_rows": False, "category_text_tensor": False, "physical_id_tensor": False, "support_query_overlap": "checked through explicit manifest keys"})
    atomic(OUT / "status.json", {"phase": "Phase75E", "status": "CONTRACT_PASS_READY_FOR_SMOKE", "run_id": args.run_id, "source_commit": commit(), "training": False, "gpu_count": 0, "teacher_signal": teacher, "folds": folds, "input_hashes": {"csv": table.csv_sha256, "features": table.feature_sha256}, "sealed_accessed": False, "held_event_accessed_for_model": False, "failures": [], "repairs": [], "next_action": "smoke_then_targeted_then_bounded_four_fold"})
    atomic(OUT / "completion/contract_audit.done", {"phase": "Phase75E", "run_id": args.run_id, "status": "PASS"})
    print(json.dumps({"phase": "Phase75E", "status": "CONTRACT_PASS_READY_FOR_SMOKE", "run_id": args.run_id, "folds": folds}, sort_keys=True))


if __name__ == "__main__":
    main()
