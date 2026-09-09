#!/usr/bin/env python3
"""Bounded Phase89 equal-budget supervisor for H3 or matched C0_REOPT."""
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
OUT = ROOT / "outputs/iclr27_phase89"
SOURCE_OUT = ROOT / "outputs/iclr27_phase88"
PY = "/home/lwr/anaconda3/envs/ovtr/bin/python"
MEMMAP = "/data2/usr_for_deadline/trackocd_phase88/shared_features"
FLOOR_KIB = int(31.25 * 1024 * 1024)
MAX_WORKERS = 4
POLL_SECONDS = 30

sys.path.insert(0, str(ROOT))
from src.iclr27_phase88.resource_manager import discover_usable_gpus, read_meminfo, safe_worker_count  # noqa: E402


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")
    os.replace(tmp, path)


def worker_rss_kib() -> int:
    vals = []
    for p in (SOURCE_OUT / "audit").glob("memory_profile_h2_known_suppression_f*.jsonl"):
        try:
            for line in p.read_text().splitlines()[-20:]:
                d = json.loads(line)
                vals.append(max(int(d.get("PSS", 0)), int(d.get("Anonymous", 0))))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return max(vals or [int(4.5 * 1024 * 1024)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=("h3_router", "c0_reopt"), required=True)
    args = ap.parse_args()
    route = args.route
    prefix = "h3_router" if route == "h3_router" else "c0_reopt"
    arch = "h3" if route == "h3_router" else "baseline"
    loss = "h3_router" if route == "h3_router" else "baseline"
    pending = []
    history = []
    completion = OUT / "completion"
    for fold in range(4):
        tag = f"{prefix}_f{fold}"
        done = completion / f"{tag}.done"
        launched = completion / f"{tag}.launched"
        if done.exists():
            history.append({"fold": fold, "tag": tag, "status": "ALREADY_DONE"})
        elif launched.exists():
            # A worker that received SIGUSR1 at the RAM safety floor leaves a
            # resumable checkpoint.  Treat that explicit marker as a pending
            # continuation; never relaunch an opaque launched unit.
            paused = completion / f"{tag}.paused_resource.json"
            if not paused.exists():
                raise RuntimeError(f"UNFINISHED_LAUNCHED_UNIT {tag}; inspect PID/checkpoint before resuming")
            payload = json.loads(paused.read_text())
            ckpt = Path(str(payload["checkpoint"]))
            if not ckpt.exists():
                raise RuntimeError(f"PAUSED_CHECKPOINT_MISSING {tag}: {ckpt}")
            pending.append({"fold": fold, "resume_checkpoint": str(ckpt),
                            "expected_start_step": int(payload["step"])})
        else:
            pending.append({"fold": fold, "resume_checkpoint": None,
                            "expected_start_step": 20000})
    preflight = {
        "phase": 89, "route": route, "architecture": arch, "loss_profile": loss,
        "started_utc": now(), "free_h": subprocess.check_output(["free", "-h"], text=True),
        "meminfo": read_meminfo(), "process_count": len(subprocess.check_output(["ps", "-e", "--no-headers"], text=True).splitlines()),
        "gpu_rows": subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True),
        "compute_apps": subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"], text=True),
        "disk": subprocess.check_output(["df", "-h", "/data1", "/data2"], text=True),
        "worker_private_kib_estimate": worker_rss_kib(), "pending_folds": pending,
        "base_checkpoint_family": str((SOURCE_OUT / "checkpoints/c0v2_fix2_formal_f{fold}.pt").resolve()),
        "start_step": 20000, "final_step": 30000, "public_dev_q1_sealed_accessed": False,
    }
    atomic_json(OUT / "audit" / f"{prefix}_preflight.json", preflight)
    active = {}
    bad_samples = 0
    paused = False

    def status(state: str) -> dict:
        clean = {int(f): {k: v for k, v in x.items() if k != "proc"} for f, x in active.items()}
        return {"phase": 89, "route": route, "task_status": state, "active_workers": clean,
                "pending_folds": pending, "worker_history": history, "resource_metric": "MemAvailable",
                "ram_floor_kib": FLOOR_KIB, "max_workers": MAX_WORKERS, "public_dev_q1_sealed_accessed": False,
                "updated_utc": now()}

    resource_log = OUT / "audit" / f"{prefix}_resource.jsonl"
    while pending or active:
        own = [int(x["proc"].pid) for x in active.values()]
        info = read_meminfo()
        gpus = discover_usable_gpus(own_pids=own, allow_shared=True)
        used = {int(x["gpu"]) for x in active.values()}
        available = [g for g in gpus if g not in used]
        safe = safe_worker_count(int(info.get("MemAvailable", 0)), worker_rss_kib(), len(available), MAX_WORKERS)
        sample = {"timestamp_utc": now(), "mem_free_kib": int(info.get("MemFree", 0)),
                  "mem_available_kib": int(info.get("MemAvailable", 0)), "cached_kib": int(info.get("Cached", 0)),
                  "buffers_kib": int(info.get("Buffers", 0)), "available_gpus": available,
                  "active_gpus": sorted(used), "safe_workers": safe}
        resource_log.parent.mkdir(parents=True, exist_ok=True)
        with resource_log.open("a") as h:
            h.write(json.dumps(sample, sort_keys=True) + "\n")
        if int(info.get("MemAvailable", 0)) < FLOOR_KIB:
            bad_samples += 1
        else:
            bad_samples = 0
        if bad_samples >= 3 and active:
            paused = True
            for item in active.values():
                os.kill(int(item["proc"].pid), signal.SIGUSR1)
            atomic_json(OUT / "audit" / f"{prefix}_supervisor_status.json", status("PAUSE_REQUESTED_RESOURCE"))
        # The pause is a transient resource action.  Once all paused workers
        # have checkpointed and the RAM floor recovers, pending continuations
        # must be allowed to launch; a latched pause would strand the route.
        if not active and int(info.get("MemAvailable", 0)) >= FLOOR_KIB:
            paused = False
        if not paused and safe > 0 and pending and available:
            slots = min(MAX_WORKERS - len(active), safe, len(pending), len(available))
            for _ in range(slots):
                item_pending = pending.pop(0)
                fold = int(item_pending["fold"]); gpu = available.pop(0); tag = f"{prefix}_f{fold}"
                log = OUT / "logs" / f"{tag}.log"; log.parent.mkdir(parents=True, exist_ok=True)
                cmd = [PY, str(ROOT / "scripts/iclr27_phase88/train_controller.py"),
                       "--fold", str(fold), "--device", "cuda:0", "--updates", "30000", "--tag", tag,
                       "--seed", "88002", "--event-tag", "fix2", "--memmap-root", MEMMAP,
                       "--resume-checkpoint", str(item_pending["resume_checkpoint"] or (SOURCE_OUT / "checkpoints" / f"c0v2_fix2_formal_f{fold}.pt")),
                       "--expected-start-step", str(item_pending["expected_start_step"]), "--checkpoint-interval", "2000",
                       "--loss-profile", loss, "--architecture", arch, "--reinit-optimizer"]
                if item_pending["resume_checkpoint"]:
                    cmd.append("--resume-launched")
                env = dict(os.environ); env.update({"TRACKOCD_OUT": str(OUT), "CUDA_VISIBLE_DEVICES": str(gpu),
                    "TRACKOCD_TORCH_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "1"})
                handle = log.open("a"); proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env); handle.close()
                active[fold] = {"fold": fold, "gpu": gpu, "tag": tag, "pid": proc.pid, "proc": proc, "command": cmd, "started_utc": now()}
        for fold, item in list(active.items()):
            code = item["proc"].poll()
            if code is not None:
                record = {k: v for k, v in item.items() if k != "proc"}
                done_path = completion / f"{item['tag']}.done"
                paused_path = completion / f"{item['tag']}.paused_resource.json"
                is_done = done_path.exists()
                record.update({"returncode": int(code), "done": is_done, "finished_utc": now()})
                history.append(record); del active[fold]
                if not is_done and paused_path.exists():
                    # The worker exited cleanly after a resource pause.  Put
                    # the exact saved step back in the bounded queue instead
                    # of treating a transient pause as route completion.
                    payload = json.loads(paused_path.read_text())
                    ckpt = Path(str(payload["checkpoint"]))
                    if not ckpt.exists():
                        raise RuntimeError(f"PAUSED_CHECKPOINT_MISSING_AFTER_EXIT {item['tag']}: {ckpt}")
                    if not any(int(x["fold"]) == int(fold) for x in pending):
                        pending.append({"fold": int(fold), "resume_checkpoint": str(ckpt),
                                        "expected_start_step": int(payload["step"])})
        atomic_json(OUT / "audit" / f"{prefix}_supervisor_status.json", status("PAUSED_RESOURCE" if paused else "IN_PROGRESS"))
        if pending or active:
            time.sleep(POLL_SECONDS)
    final_state = "PAUSED_RESOURCE" if paused else ("DONE" if all((completion / f"{prefix}_f{f}.done").exists() for f in range(4)) else "FAILED")
    final = status(final_state); final["completed_utc"] = now()
    atomic_json(OUT / "audit" / f"{prefix}_supervisor_status.json", final)
    print(json.dumps(final, indent=2, sort_keys=True, default=str))
    return 0 if final_state == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
