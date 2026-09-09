#!/usr/bin/env python3
"""Bounded four-fold TRAIN-disjoint validation for H2_EQUAL_BUDGET."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"
PY = "/home/lwr/anaconda3/envs/ovtr/bin/python"
MEMMAP = "/data2/usr_for_deadline/trackocd_phase88/shared_features"

import sys
sys.path.insert(0, str(ROOT))
from src.iclr27_phase88.resource_manager import discover_usable_gpus, read_meminfo


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> int:
    pre = {"phase": 88, "route": "H2_EQUAL_BUDGET", "split": "TRAIN_disjoint_validation",
           "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "meminfo": read_meminfo(), "gpu_rows": subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)}
    atomic_json(OUT / "audit/h2_equal_validation_preflight.json", pre)
    procs = {}
    history = []
    pending = list(range(4))
    while pending or procs:
        used = {int(x["gpu"]) for x in procs.values()}
        gpus = [x for x in discover_usable_gpus(own_pids=[int(v["proc"].pid) for v in procs.values()], allow_shared=False) if x not in used]
        info = read_meminfo()
        if int(info.get("MemAvailable", 0)) > int(31.25 * 1024 * 1024):
            while pending and gpus and len(procs) < 4:
                fold = pending.pop(0); gpu = gpus.pop(0)
                tag = f"h2_equal_f{fold}_val"
                root = OUT / "validation" / tag
                if (root / "final.done").exists():
                    history.append({"fold": fold, "tag": tag, "status": "ALREADY_DONE"}); continue
                log = OUT / "logs" / f"{tag}.log"; log.parent.mkdir(parents=True, exist_ok=True)
                cmd = [PY, str(ROOT / "scripts/iclr27_phase88/evaluate_checkpoint_sharded.py"),
                       "--fold", str(fold), "--checkpoint", str(OUT / "checkpoints" / f"h2_equal_f{fold}.pt"),
                       "--tag", tag, "--event-tag", "fix2", "--device", "cuda:0", "--shard-size", "250", "--memmap-root", MEMMAP]
                env = dict(os.environ); env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "TRACKOCD_TORCH_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                handle = log.open("a"); proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env); handle.close()
                procs[fold] = {"fold": fold, "gpu": gpu, "tag": tag, "pid": proc.pid, "proc": proc, "command": cmd, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        for fold, item in list(procs.items()):
            code = item["proc"].poll()
            if code is not None:
                finished = {k: v for k, v in item.items() if k != "proc"}
                finished.update({"returncode": int(code), "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "done": (OUT / "validation" / item["tag"] / "final.done").exists()})
                history.append(finished)
                del procs[fold]
        atomic_json(OUT / "audit/h2_equal_validation_status.json", {"phase": 88, "route": "H2_EQUAL_BUDGET", "active": {str(k): {x:y for x,y in v.items() if x != "proc"} for k,v in procs.items()}, "pending": pending, "history": history, "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        if pending or procs: time.sleep(15)
    status = "DONE" if all((OUT / "validation" / f"h2_equal_f{f}_val" / "final.done").exists() for f in range(4)) else "FAILED"
    atomic_json(OUT / "audit/h2_equal_validation_status.json", {"phase": 88, "route": "H2_EQUAL_BUDGET", "status": status, "pending": pending, "history": history, "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    print(json.dumps({"status": status, "history": history}, indent=2, default=str))
    return 0 if status == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
