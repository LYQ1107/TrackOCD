#!/usr/bin/env python3
"""Bounded TRAIN-disjoint validation for Phase90 reproduction routes."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase90"
PY = "/home/lwr/anaconda3/envs/ovtr/bin/python"
MEMMAP = "/data2/usr_for_deadline/trackocd_phase88/shared_features"
GPU_POOL = [1, 3, 7, 8]
FLOOR_KIB = int(31.25 * 1024 * 1024)


def now() -> str: return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")
    os.replace(tmp, path)


def free_gpus(used: set[int]) -> list[int]:
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in raw.splitlines():
        v = [x.strip() for x in line.split(",")]
        if len(v) >= 3 and int(v[0]) in GPU_POOL and int(v[0]) not in used and int(v[2]) > 1024:
            rows.append(int(v[0]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--route", choices=("h3_repro", "c0_reopt_repro"), required=True); args = ap.parse_args()
    route = args.route; arch = "h3" if route == "h3_repro" else "baseline"
    pending = [f for f in range(4) if not (OUT / "validation" / f"{route}_f{f}_val/final.done").exists()]
    active = {}; history = []
    atomic_json(OUT / "audit" / f"{route}_validation_preflight.json", {"phase": 90, "route": route, "split": "TRAIN_disjoint_fix2_validation", "architecture": arch, "gpu_pool": GPU_POOL, "pending": pending, "mem_available_kib": int(next((x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')), 0)), "public_dev_q1_sealed_accessed": False, "started_utc": now()})
    while pending or active:
        used = {int(x["gpu"]) for x in active.values()}
        gpus = free_gpus(used)
        mem_avail = int(next((x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')), 0))
        while pending and gpus and mem_avail > FLOOR_KIB and len(active) < 4:
            fold = pending.pop(0); gpu = gpus.pop(0); tag = f"{route}_f{fold}_val"
            log = OUT / "logs" / f"{tag}.log"; log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [PY, str(ROOT / "scripts/iclr27_phase88/evaluate_checkpoint_sharded.py"), "--fold", str(fold), "--checkpoint", str((OUT / "checkpoints" / f"{route}_f{fold}.pt").resolve()), "--tag", tag, "--event-tag", "fix2", "--device", "cuda:0", "--shard-size", "250", "--memmap-root", MEMMAP, "--architecture", arch]
            env = dict(os.environ); env.update({"TRACKOCD_OUT": str(OUT.resolve()), "CUDA_VISIBLE_DEVICES": str(gpu), "TRACKOCD_TORCH_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            h = log.open("a"); proc = subprocess.Popen(cmd, stdout=h, stderr=subprocess.STDOUT, env=env); h.close()
            active[fold] = {"fold": fold, "gpu": gpu, "tag": tag, "pid": proc.pid, "proc": proc, "command": cmd, "started_utc": now()}
        for fold, item in list(active.items()):
            code = item["proc"].poll()
            if code is not None:
                rec = {k: v for k, v in item.items() if k != "proc"}; rec.update({"returncode": int(code), "done": (OUT / "validation" / item["tag"] / "final.done").exists(), "finished_utc": now()}); history.append(rec); del active[fold]
                if not rec["done"]:
                    raise RuntimeError(f"PHASE90_VALIDATION_FAILED {item['tag']} returncode={code}; inspect log")
        atomic_json(OUT / "audit" / f"{route}_validation_status.json", {"phase": 90, "route": route, "active": {str(k): {x:y for x,y in v.items() if x != "proc"} for k,v in active.items()}, "pending": pending, "history": history, "public_dev_q1_sealed_accessed": False, "updated_utc": now()})
        if pending or active: time.sleep(30)
    status = "DONE" if all((OUT / "validation" / f"{route}_f{f}_val/final.done").exists() for f in range(4)) else "FAILED"
    atomic_json(OUT / "audit" / f"{route}_validation_status.json", {"phase": 90, "route": route, "status": status, "history": history, "completed_utc": now(), "public_dev_q1_sealed_accessed": False})
    print(json.dumps({"phase": 90, "route": route, "status": status, "history": history}, indent=2, sort_keys=True, default=str))
    return 0 if status == "DONE" else 1


if __name__ == "__main__": raise SystemExit(main())
