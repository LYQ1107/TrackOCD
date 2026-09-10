#!/usr/bin/env python3
"""Bounded Phase90 H3/C0 equal-budget supervisor.

The first launch from the frozen 20k C0 checkpoint intentionally creates a
fresh AdamW state.  A continuation after an explicit RAM pause restores the
optimizer and all sampler/RNG state and never passes ``--reinit-optimizer``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase90"
SOURCE_OUT = ROOT / "outputs/iclr27_phase88"
PY = "/home/lwr/anaconda3/envs/ovtr/bin/python"
MEMMAP = "/data2/usr_for_deadline/trackocd_phase88/shared_features"
GPU_POOL = [1, 3, 7, 8]
FLOOR_KIB = int(31.25 * 1024 * 1024)
MAX_WORKERS = 4
POLL_SECONDS = 30

sys.path.insert(0, str(ROOT))
from src.iclr27_phase88.resource_manager import read_meminfo, safe_worker_count  # noqa: E402


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")
    os.replace(tmp, path)


def gpu_rows() -> list[dict[str, int]]:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in raw.splitlines():
        vals = [x.strip() for x in line.split(",")]
        if len(vals) >= 4 and int(vals[0]) in GPU_POOL:
            rows.append({"index": int(vals[0]), "used_mib": int(vals[1]), "free_mib": int(vals[2]), "util": int(vals[3])})
    return rows


def free_pool(used: set[int]) -> list[int]:
    # Do not assign a card with a foreign process.  The route is intentionally
    # restricted to the preflight-idle pool so external jobs are untouched.
    try:
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    # UUID-to-index mapping is read-only and robust to MIG-disabled A100s.
    mapping_raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    mapping = {row.split(",", 1)[1].strip(): int(row.split(",", 1)[0].strip()) for row in mapping_raw.splitlines() if "," in row}
    occupied: set[int] = set()
    for line in apps.splitlines():
        vals = [x.strip() for x in line.split(",")]
        if vals and vals[0] in mapping and mapping[vals[0]] in GPU_POOL:
            occupied.add(mapping[vals[0]])
    return [row["index"] for row in gpu_rows() if row["index"] not in occupied and row["index"] not in used and row["free_mib"] > 1024]


def worker_rss_kib() -> int:
    # Phase89 measured ~4.5 GiB PSS; use a conservative 6 GiB budget for a
    # Phase90 worker so the 25% RAM floor remains enforceable.
    return int(6 * 1024 * 1024)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=("h3_repro", "c0_reopt_repro"), required=True)
    args = ap.parse_args()
    route = args.route
    prefix = route
    arch = "h3" if route == "h3_repro" else "baseline"
    loss = "h3_router" if route == "h3_repro" else "baseline"
    completion = OUT / "completion"
    pending: list[dict] = []
    history: list[dict] = []
    for fold in range(4):
        tag = f"{prefix}_f{fold}"
        done = completion / f"{tag}.done"
        launched = completion / f"{tag}.launched"
        paused = completion / f"{tag}.paused_resource.json"
        if done.exists():
            history.append({"fold": fold, "tag": tag, "status": "ALREADY_DONE"})
        elif launched.exists():
            if not paused.exists():
                raise RuntimeError(f"UNFINISHED_LAUNCHED_UNIT {tag}; no explicit Phase90 pause record")
            p = json.loads(paused.read_text())
            ckpt = Path(str(p["checkpoint"]))
            if not ckpt.exists():
                raise RuntimeError(f"PAUSED_CHECKPOINT_MISSING {tag}: {ckpt}")
            pending.append({"fold": fold, "resume_checkpoint": str(ckpt), "expected_start_step": int(p["step"])})
        else:
            base = SOURCE_OUT / "checkpoints" / f"c0v2_fix2_formal_f{fold}.pt"
            if not base.exists():
                raise FileNotFoundError(base)
            pending.append({"fold": fold, "resume_checkpoint": None, "expected_start_step": 20000})
    preflight = {
        "phase": 90, "route": route, "architecture": arch, "loss_profile": loss,
        "started_utc": now(), "free_h": subprocess.check_output(["free", "-h"], text=True),
        "meminfo": read_meminfo(), "process_count": len(subprocess.check_output(["ps", "-e", "--no-headers"], text=True).splitlines()),
        "gpu_rows": gpu_rows(), "gpu_pool": GPU_POOL,
        "disk": subprocess.check_output(["df", "-h", "/data1", "/data2"], text=True),
        "worker_private_kib_estimate": worker_rss_kib(), "pending_folds": pending,
        "base_step": 20000, "final_step": 30000, "reinit_only_initial_launch": True,
        "public_dev_q1_sealed_accessed": False,
    }
    atomic_json(OUT / "audit" / f"{prefix}_preflight.json", preflight)
    active: dict[int, dict] = {}
    low_samples = 0
    resource_log = OUT / "audit" / f"{prefix}_resource.jsonl"

    def write_status(state: str) -> None:
        clean = {str(f): {k: v for k, v in item.items() if k != "proc"} for f, item in active.items()}
        atomic_json(OUT / "audit" / f"{prefix}_supervisor_status.json", {
            "phase": 90, "route": route, "task_status": state, "active_workers": clean,
            "pending_folds": pending, "worker_history": history, "resource_metric": "MemAvailable",
            "ram_floor_kib": FLOOR_KIB, "gpu_pool": GPU_POOL, "max_workers": MAX_WORKERS,
            "public_dev_q1_sealed_accessed": False, "updated_utc": now(),
        })

    while pending or active:
        info = read_meminfo()
        available = free_pool({int(x["gpu"]) for x in active.values()})
        safe = safe_worker_count(int(info.get("MemAvailable", 0)), worker_rss_kib(), len(available), MAX_WORKERS)
        sample = {"timestamp_utc": now(), "mem_available_kib": int(info.get("MemAvailable", 0)), "mem_free_kib": int(info.get("MemFree", 0)), "available_gpus": available, "active_gpus": sorted(int(x["gpu"]) for x in active.values()), "safe_workers": safe}
        resource_log.parent.mkdir(parents=True, exist_ok=True)
        with resource_log.open("a") as handle:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        low_samples = low_samples + 1 if int(info.get("MemAvailable", 0)) < FLOOR_KIB else 0
        if low_samples >= 3 and active:
            for item in active.values():
                os.kill(int(item["proc"].pid), signal.SIGUSR1)
            write_status("PAUSE_REQUESTED_RESOURCE")
        if safe > 0 and pending and available and low_samples == 0:
            slots = min(MAX_WORKERS - len(active), safe, len(pending), len(available))
            for _ in range(slots):
                item = pending.pop(0); fold = int(item["fold"]); gpu = available.pop(0); tag = f"{prefix}_f{fold}"
                log = OUT / "logs" / f"{tag}.log"; log.parent.mkdir(parents=True, exist_ok=True)
                base = SOURCE_OUT / "checkpoints" / f"c0v2_fix2_formal_f{fold}.pt"
                cmd = [PY, str(ROOT / "scripts/iclr27_phase88/train_controller.py"), "--fold", str(fold), "--device", "cuda:0", "--updates", "30000", "--tag", tag, "--seed", "88002", "--event-tag", "fix2", "--memmap-root", MEMMAP, "--resume-checkpoint", str(item["resume_checkpoint"] or base), "--expected-start-step", str(item["expected_start_step"]), "--checkpoint-interval", "2000", "--loss-profile", loss, "--architecture", arch]
                if item["resume_checkpoint"] is None:
                    cmd.append("--reinit-optimizer")
                else:
                    cmd.append("--resume-launched")
                env = dict(os.environ); env.update({"TRACKOCD_OUT": str(OUT.resolve()), "CUDA_VISIBLE_DEVICES": str(gpu), "TRACKOCD_TORCH_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "1"})
                handle = log.open("a")
                proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
                handle.close()
                active[fold] = {"fold": fold, "gpu": gpu, "tag": tag, "pid": proc.pid, "proc": proc, "command": cmd, "started_utc": now(), "resume_mode": "RESTORE_AFTER_PAUSE" if item["resume_checkpoint"] else "REINIT_ONCE_AT_BASE"}
        for fold, item in list(active.items()):
            code = item["proc"].poll()
            if code is not None:
                rec = {k: v for k, v in item.items() if k != "proc"}; rec.update({"returncode": int(code), "done": (completion / f"{item['tag']}.done").exists(), "finished_utc": now()})
                history.append(rec); del active[fold]
                paused = completion / f"{item['tag']}.paused_resource.json"
                if not rec["done"] and paused.exists():
                    p = json.loads(paused.read_text()); ckpt = Path(str(p["checkpoint"]))
                    if not ckpt.exists():
                        raise RuntimeError(f"PAUSED_CHECKPOINT_MISSING_AFTER_EXIT {item['tag']}: {ckpt}")
                    pending.append({"fold": fold, "resume_checkpoint": str(ckpt), "expected_start_step": int(p["step"])})
                elif not rec["done"]:
                    raise RuntimeError(f"PHASE90_WORKER_FAILED {item['tag']} returncode={code}; inspect log")
        write_status("PAUSED_RESOURCE" if low_samples >= 3 else "IN_PROGRESS")
        if pending or active:
            time.sleep(POLL_SECONDS)
    final_state = "DONE" if all((completion / f"{prefix}_f{fold}.done").exists() for fold in range(4)) else "FAILED"
    write_status(final_state)
    print(json.dumps({"phase": 90, "route": route, "status": final_state, "history": history}, indent=2, sort_keys=True, default=str))
    return 0 if final_state == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
