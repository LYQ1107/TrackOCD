#!/usr/bin/env python3
"""Bounded, completion-based Phase88R equal-budget H2 supervisor.

The supervisor schedules at most four fold workers on GPUs discovered from
read-only nvidia-smi output.  It uses MemAvailable for safety decisions and
keeps MemFree only in the audit log.  It never touches external PIDs.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"
PY = "/home/lwr/anaconda3/envs/ovtr/bin/python"
MEMMAP = "/data2/usr_for_deadline/trackocd_phase88/shared_features"
FLOOR_KIB = int(31.25 * 1024 * 1024)
MAX_WORKERS = 4
POLL_SECONDS = 30
EXPECTED_START = 20000
FINAL_STEP = 30000

sys.path.insert(0, str(ROOT))
from src.iclr27_phase88.resource_manager import (  # noqa: E402
    discover_usable_gpus,
    process_memory_detail,
    read_meminfo,
    safe_worker_count,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(value, sort_keys=True) + "\n")
        f.flush()


def measured_private_kib() -> int:
    vals = []
    for p in (OUT / "audit").glob("memory_profile_h2_known_suppression_f*.jsonl"):
        try:
            for line in p.read_text().splitlines()[-20:]:
                row = json.loads(line)
                vals.append(max(int(row.get("PSS", 0)), int(row.get("Anonymous", 0))))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    # The old CUDA formal profile measured ~4.4 GiB PSS and is the conservative
    # single-worker estimate for the shared-memmap route.
    return max(vals or [int(4.5 * 1024 * 1024)])


def status_payload(active: dict, pending: list[int], history: list[dict], state: str = "IN_PROGRESS") -> dict:
    clean_active = {
        int(fold): {k: v for k, v in item.items() if k != "proc"}
        for fold, item in active.items()
    }
    return {
        "schema_version": "trackocd.phase88.h2_equal_supervisor.v1",
        "phase": 88,
        "route": "H2_EQUAL_BUDGET",
        "task_status": state,
        "active_workers": clean_active,
        "pending_folds": pending,
        "worker_history": history,
        "resource_metric": "MemAvailable",
        "ram_floor_kib": FLOOR_KIB,
        "max_workers": MAX_WORKERS,
        "public_dev_q1_sealed_accessed": False,
        "updated_utc": now(),
    }


def main() -> int:
    folds = list(range(4))
    completion = OUT / "completion"
    active: dict[int, dict] = {}
    pending = []
    history: list[dict] = []
    for fold in folds:
        tag = f"h2_equal_f{fold}"
        done = completion / f"{tag}.done"
        launched = completion / f"{tag}.launched"
        if done.exists():
            history.append({"fold": fold, "tag": tag, "status": "ALREADY_DONE"})
        elif launched.exists():
            # Never blindly relaunch a unit whose marker says it was spawned.
            raise RuntimeError(f"UNFINISHED_LAUNCHED_UNIT {tag}; inspect its recorded PID before resume")
        else:
            pending.append(fold)

    preflight = {
        "phase": 88,
        "route": "H2_EQUAL_BUDGET",
        "started_utc": now(),
        "free_h": subprocess.check_output(["free", "-h"], text=True),
        "meminfo": read_meminfo(),
        "process_count": len(subprocess.check_output(["ps", "-e", "--no-headers"], text=True).splitlines()),
        "gpu_rows": subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True),
        "compute_apps": subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"], text=True),
        "disk": subprocess.check_output(["df", "-h", "/data1", "/data2"], text=True),
        "measured_worker_private_kib": measured_private_kib(),
        "pending_folds": pending,
    }
    atomic_json(OUT / "audit/h2_equal_preflight.json", preflight)
    atomic_json(OUT / "audit/h2_equal_supervisor_status.json", status_payload({}, pending, history))
    resource_log = OUT / "audit/h2_equal_resource.jsonl"
    bad_samples = 0
    paused = False

    while pending or active:
        own_pids = [int(x["proc"].pid) for x in active.values()]
        info = read_meminfo()
        private = measured_private_kib()
        gpus = discover_usable_gpus(own_pids=own_pids, allow_shared=False)
        used_gpus = {int(x["gpu"]) for x in active.values()}
        available_gpus = [g for g in gpus if g not in used_gpus]
        safe = safe_worker_count(int(info.get("MemAvailable", 0)), private, len(available_gpus), MAX_WORKERS)
        sample = {
            "timestamp_utc": now(), "mem_free_kib": int(info.get("MemFree", 0)),
            "mem_available_kib": int(info.get("MemAvailable", 0)),
            "cached_kib": int(info.get("Cached", 0)), "buffers_kib": int(info.get("Buffers", 0)),
            "worker_private_kib": private, "available_gpus": available_gpus,
            "active_gpus": sorted(used_gpus), "safe_workers": safe,
        }
        append_jsonl(resource_log, sample)
        if int(info.get("MemAvailable", 0)) < FLOOR_KIB:
            bad_samples += 1
        else:
            bad_samples = 0
        if bad_samples >= 3 and active:
            paused = True
            for item in active.values():
                os.kill(int(item["proc"].pid), signal.SIGUSR1)
            atomic_json(OUT / "audit/h2_equal_supervisor_status.json", status_payload(active, pending, history, "PAUSE_REQUESTED_RESOURCE"))

        if not paused and safe > 0 and pending and available_gpus:
            slots = min(MAX_WORKERS - len(active), safe, len(pending), len(available_gpus))
            for _ in range(slots):
                fold = pending.pop(0)
                gpu = available_gpus.pop(0)
                tag = f"h2_equal_f{fold}"
                log = OUT / "logs" / f"{tag}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                cmd = [PY, str(ROOT / "scripts/iclr27_phase88/train_controller.py"),
                       "--fold", str(fold), "--device", "cuda:0", "--updates", str(FINAL_STEP),
                       "--tag", tag, "--seed", "88002", "--event-tag", "fix2",
                       "--memmap-root", MEMMAP, "--resume-checkpoint",
                       str(OUT / "checkpoints" / f"c0v2_fix2_formal_f{fold}.pt"),
                       "--expected-start-step", str(EXPECTED_START), "--checkpoint-interval", "2000",
                       "--loss-profile", "h2_known_suppression"]
                env = dict(os.environ)
                env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "TRACKOCD_TORCH_THREADS": "1",
                            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "1"})
                handle = log.open("a")
                proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
                handle.close()
                active[fold] = {"fold": fold, "gpu": gpu, "tag": tag, "pid": proc.pid, "proc": proc, "command": cmd, "started_utc": now()}
        for fold, item in list(active.items()):
            proc = item["proc"]
            code = proc.poll()
            if code is not None:
                history.append({k: v for k, v in item.items() if k != "proc"})
                history[-1].update({"returncode": int(code), "done": (completion / f"{item['tag']}.done").exists(), "finished_utc": now()})
                del active[fold]
        atomic_json(OUT / "audit/h2_equal_supervisor_status.json", status_payload(active, pending, history, "PAUSED_RESOURCE" if paused else "IN_PROGRESS"))
        if paused:
            break
        if pending or active:
            time.sleep(POLL_SECONDS)

    final_state = "PAUSED_RESOURCE" if paused else ("DONE" if not pending and not active and all((completion / f"h2_equal_f{f}.done").exists() for f in folds) else "FAILED")
    final = status_payload(active, pending, history, final_state)
    final["completed_utc"] = now()
    atomic_json(OUT / "audit/h2_equal_supervisor_status.json", final)
    print(json.dumps(final, indent=2, sort_keys=True, default=str))
    return 0 if final_state == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
