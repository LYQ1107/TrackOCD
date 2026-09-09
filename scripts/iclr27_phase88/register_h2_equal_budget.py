#!/usr/bin/env python3
"""Register the Phase88R equal-budget H2 route without touching old artifacts."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"
CKPT = OUT / "checkpoints"


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
    import torch

    folds = []
    for fold in range(4):
        path = CKPT / f"c0v2_fix2_formal_f{fold}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu")
        checks = {
            "fold": int(payload.get("fold", -1)),
            "step": int(payload.get("step", -1)),
            "route": payload.get("route"),
            "event_tag": payload.get("event_tag"),
            "loss_profile": payload.get("loss_profile"),
            "optimizer_state": payload.get("optimizer") is not None,
            "sampler_state": payload.get("sampler_state") is not None,
            "rollout_rng_state": payload.get("rollout_rng_state") is not None,
            "python_rng_state": payload.get("python_rng_state") is not None,
            "numpy_rng_state": payload.get("numpy_rng_state") is not None,
            "torch_rng_state": payload.get("torch_rng_state") is not None,
            "sha256": sha(path),
            "path": str(path.resolve()),
        }
        if checks["fold"] != fold or checks["step"] != 20000:
            raise RuntimeError(f"invalid equal-budget base: {checks}")
        # Older C0 fix2 finals predate the explicit loss_profile field; None
        # is the historical baseline default and is retained as provenance.
        if checks["event_tag"] != "fix2" or checks["loss_profile"] not in (None, "baseline"):
            raise RuntimeError(f"base contract mismatch: {checks}")
        checks["loss_profile_effective"] = checks["loss_profile"] or "baseline (implicit legacy default)"
        checks["resume_state_complete"] = all(checks[k] for k in ("optimizer_state", "sampler_state", "rollout_rng_state", "python_rng_state", "numpy_rng_state", "torch_rng_state"))
        checks["resume_state_repair_required"] = not checks["resume_state_complete"]
        folds.append(checks)

    resource = {
        "free_h": subprocess.check_output(["free", "-h"], text=True),
        "meminfo": subprocess.check_output(["grep", "-E", "MemTotal|MemFree|MemAvailable|Cached|Buffers|Swap", "/proc/meminfo"], text=True),
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True),
        "process_count": len(subprocess.check_output(["ps", "-e", "--no-headers"], text=True).splitlines()),
        "disk": subprocess.check_output(["df", "-h", "/data1", "/data2"], text=True),
    }
    prereg = {
        "schema_version": "trackocd.phase88.h2_equal_budget_preregistration.v1",
        "phase": 88,
        "route": "H2_EQUAL_BUDGET",
        "status": "REGISTERED",
        "base_step": 20000,
        "final_step": 30000,
        "additional_updates": 10000,
        "folds": 4,
        "seed": 88002,
        "support_mode": False,
        "event_tag": "fix2",
        "comparison": "c0_continue_fix1",
        "only_difference": "known_suppression_loss_weight_1.5",
        "loss_profile": "h2_known_suppression",
        "resume_contract": {"optimizer": True, "sampler": True, "python_rng": True, "numpy_rng": True, "torch_rng": True, "rollout_rng": True, "expected_start_step": 20000,
                            "legacy_missing_state": "f0 checkpoint predates sampler/rollout persistence; deterministic sampler initialization is recorded rather than silently presented as exact continuation"},
        "base_checkpoints": folds,
        "selection": "mean exact Phase19R TRAIN-disjoint selection_score across four folds; H2 selected only if strictly greater than C0",
        "held_or_public_selection": False,
        "resource_metric": "MemAvailable",
        "ram_floor_gib": 31.25,
        "resource_breach_rule": "3 consecutive MemAvailable samples 30 seconds apart; MemFree diagnostic only",
        "resource_preflight": resource,
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_or_text_as_model_input": False,
        "registered_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(OUT / "audit/hypothesis_2_equal_budget_preregistration.json", prereg)
    print(json.dumps(prereg, indent=2, sort_keys=True))

    correction = {
        "schema_version": "trackocd.phase88.resource_metric_correction.v1",
        "status": "RECLASSIFIED_NOT_DELETED",
        "formal_safety_metric": "MemAvailable",
        "ram_floor_gib": 31.25,
        "historical_stop_criterion": "MemFree",
        "historical_stop_criterion_valid": False,
        "reason": "MemFree is sensitive to reclaimable page cache in shared NumPy memmap workloads; MemAvailable is the kernel safety estimate.",
        "historical_artifacts": {},
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    for name in ("h2_resource_stop.json", "h2_resume2_resource_stop.json", "h2_cpu_parallel_resource_stop.json", "h2_cpu_sequential_resource_stop.json"):
        p = OUT / "audit" / name
        if p.exists():
            try:
                old = json.loads(p.read_text())
            except json.JSONDecodeError:
                old = {"parseable": False}
            correction["historical_artifacts"][name] = {
                "path": str(p.resolve()),
                "mem_free_kib": old.get("mem_free_before_stop_kb"),
                "mem_available_kib": old.get("mem_available_before_stop_kb"),
                "old_status": old.get("status"),
                "classification": "HISTORICAL_MEMFREE_TRIGGER_RECLASSIFIED",
            }
    atomic_json(OUT / "audit/resource_metric_correction.json", correction)
    atomic_json(OUT / "audit/continuous_state.json", {
        "phase": 88,
        "task_status": "IN_PROGRESS",
        "current_stage": "H2_EQUAL_BUDGET",
        "resource_blocker": False,
        "resource_metric": "MemAvailable",
        "pending": ["H2 equal-budget four-fold train", "H2 four-fold TRAIN validation", "H2 TRAIN selection", "held diagnostic if selected"],
        "public_dev_q1_sealed_accessed": False,
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
