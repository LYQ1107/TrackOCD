#!/usr/bin/env python3
"""Bounded TRAIN-disjoint validation for one Phase89 route."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase89"
PY = "/home/lwr/anaconda3/envs/ovtr/bin/python"
MEMMAP = "/data2/usr_for_deadline/trackocd_phase88/shared_features"
FLOOR_KIB = int(31.25 * 1024 * 1024)

import sys
sys.path.insert(0, str(ROOT))
from src.iclr27_phase88.resource_manager import discover_usable_gpus, read_meminfo  # noqa: E402


def now() -> str: return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=("h3_router", "c0_reopt"), required=True)
    args = ap.parse_args()
    prefix = args.route
    arch = "h3" if prefix == "h3_router" else "baseline"
    pending = []
    history = []
    for fold in range(4):
        tag = f"{prefix}_f{fold}_val"
        root = OUT / "validation" / tag
        if (root / "final.done").exists(): history.append({"fold": fold, "status": "ALREADY_DONE"})
        else: pending.append(fold)
    active = {}
    pre = {"phase": 89, "route": prefix, "split": "TRAIN_disjoint_validation", "architecture": arch,
           "started_utc": now(), "meminfo": read_meminfo(),
           "gpu_rows": subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True),
           "pending": pending, "public_dev_q1_sealed_accessed": False}
    atomic_json(OUT / "audit" / f"{prefix}_validation_preflight.json", pre)
    while pending or active:
        own = [int(x["proc"].pid) for x in active.values()]
        used = {int(x["gpu"]) for x in active.values()}
        gpus = [g for g in discover_usable_gpus(own_pids=own, allow_shared=True) if g not in used]
        mem_ok = int(read_meminfo().get("MemAvailable", 0)) > FLOOR_KIB
        while pending and gpus and mem_ok and len(active) < 4:
            fold = pending.pop(0); gpu = gpus.pop(0); tag = f"{prefix}_f{fold}_val"
            log = OUT / "logs" / f"{tag}.log"; log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [PY, str(ROOT / "scripts/iclr27_phase88/evaluate_checkpoint_sharded.py"),
                   "--fold", str(fold), "--checkpoint", str(OUT / "checkpoints" / f"{prefix}_f{fold}.pt"),
                   "--tag", tag, "--event-tag", "fix2", "--device", "cuda:0", "--shard-size", "250",
                   "--memmap-root", MEMMAP, "--architecture", arch]
            env = dict(os.environ); env.update({"TRACKOCD_OUT": str(OUT), "CUDA_VISIBLE_DEVICES": str(gpu),
                "TRACKOCD_TORCH_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "1"})
            h = log.open("a"); proc = subprocess.Popen(cmd, stdout=h, stderr=subprocess.STDOUT, env=env); h.close()
            active[fold] = {"fold": fold, "gpu": gpu, "tag": tag, "pid": proc.pid, "proc": proc, "command": cmd, "started_utc": now()}
        for fold, item in list(active.items()):
            code = item["proc"].poll()
            if code is not None:
                rec = {k: v for k, v in item.items() if k != "proc"}; rec.update({"returncode": int(code), "finished_utc": now(), "done": (OUT / "validation" / item["tag"] / "final.done").exists()})
                history.append(rec); del active[fold]
        atomic_json(OUT / "audit" / f"{prefix}_validation_status.json", {"phase": 89, "route": prefix, "active": {str(k): {x:y for x,y in v.items() if x != "proc"} for k,v in active.items()}, "pending": pending, "history": history, "updated_utc": now()})
        if pending or active: time.sleep(30)
    status = "DONE" if all((OUT / "validation" / f"{prefix}_f{f}_val" / "final.done").exists() for f in range(4)) else "FAILED"
    atomic_json(OUT / "audit" / f"{prefix}_validation_status.json", {"phase": 89, "route": prefix, "status": status, "history": history, "completed_utc": now()})
    print(json.dumps({"status": status, "history": history}, indent=2, sort_keys=True, default=str))
    return 0 if status == "DONE" else 1


if __name__ == "__main__": raise SystemExit(main())
