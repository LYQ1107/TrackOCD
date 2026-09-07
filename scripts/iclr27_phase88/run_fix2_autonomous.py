#!/usr/bin/env python3
"""Bounded Phase88 fix2 supervisor with measured RSS-based concurrency."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"
PY = "/home/lwr/anaconda3/envs/ovtr/bin/python"
SHARED = "/data2/usr_for_deadline/trackocd_phase88/shared_features"
GPU = 5
FLOOR_KIB = int(31.25 * 1024 * 1024)
MEASURED_RSS_KIB = int(5052000)  # measured fix2 targeted peak, overwritten from artifact when present


def registration() -> dict:
    return json.loads((OUT / "audit/window_registration.json").read_text())


def remaining_seconds() -> float:
    value = registration()["deadline_utc"].replace("Z", "+00:00")
    deadline = dt.datetime.fromisoformat(value).astimezone(dt.timezone.utc)
    return (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()


def available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return 0


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    if not (OUT / "completion/build_events_v2_fix2.done").exists():
        raise RuntimeError("fix2 event manifest is not complete")
    metric = OUT / "metrics/c0v2_fix2_targeted_f0.json"
    if metric.exists():
        try:
            samples = json.loads(metric.read_text()).get("rss_samples", [])
            if samples:
                global MEASURED_RSS_KIB
                MEASURED_RSS_KIB = math.ceil(max(x["rss_bytes"] for x in samples) / 1024)
        except Exception:
            pass
    headroom = available_kib() - FLOOR_KIB
    safe_workers = max(1, min(4, headroom // max(1, int(1.25 * MEASURED_RSS_KIB))))
    # GPU6-9 are occupied in the current snapshot; only GPU5 was selected.
    safe_workers = min(safe_workers, 1)
    status = {"phase": 88, "route": "C0V2_FIX2", "gpu": GPU,
              "measured_peak_rss_kib": MEASURED_RSS_KIB,
              "mem_available_kib": available_kib(), "ram_floor_kib": FLOOR_KIB,
              "safe_workers": int(safe_workers), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "folds": [], "external_processes_touched": False}
    atomic_json(OUT / "audit/fix2_supervisor_start.json", status)
    for fold in range(4):
        tag = f"c0v2_fix2_formal_f{fold}"
        done = OUT / "completion" / f"{tag}.done"
        launched = OUT / "completion" / f"{tag}.launched"
        if done.exists():
            status["folds"].append({"fold": fold, "status": "SKIP_DONE"})
            continue
        if launched.exists():
            status["folds"].append({"fold": fold, "status": "STOP_LAUNCHED_WITHOUT_DONE"})
            break
        if available_kib() < FLOOR_KIB:
            status["folds"].append({"fold": fold, "status": "STOP_RAM_FLOOR", "available_kib": available_kib()})
            break
        train_log = OUT / "logs" / f"{tag}.log"
        train_log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(ROOT / "scripts/iclr27_phase88/train_controller.py"),
               "--fold", str(fold), "--device", f"cuda:{GPU}", "--updates", "20000",
               "--tag", tag, "--checkpoint-interval", "2000", "--event-tag", "fix2",
               "--memmap-root", SHARED]
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        with train_log.open("w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        entry = {"fold": fold, "train_returncode": proc.returncode, "train_started_utc": started,
                 "train_finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "available_after_train_kib": available_kib()}
        if proc.returncode != 0:
            entry["status"] = "TRAIN_FAILED"
            status["folds"].append(entry)
            atomic_json(OUT / "audit/fix2_supervisor_status.json", status)
            raise SystemExit(proc.returncode)
        val_tag = f"{tag}_val"
        val_log = OUT / "logs" / f"{val_tag}.log"
        val_cmd = [PY, str(ROOT / "scripts/iclr27_phase88/evaluate_checkpoint_sharded.py"),
                   "--fold", str(fold), "--checkpoint", str(OUT / "checkpoints" / f"{tag}.pt"),
                   "--tag", val_tag, "--event-tag", "fix2", "--shard-size", "250",
                   "--device", f"cuda:{GPU}", "--memmap-root", SHARED]
        with val_log.open("w") as f:
            val_proc = subprocess.run(val_cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        entry.update({"validation_returncode": val_proc.returncode,
                      "validation_finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "status": "DONE" if val_proc.returncode == 0 else "VALIDATION_FAILED"})
        status["folds"].append(entry)
        atomic_json(OUT / "audit/fix2_supervisor_status.json", status)
        if val_proc.returncode != 0:
            raise SystemExit(val_proc.returncode)
        if remaining_seconds() <= 2700:
            status["stop_reason"] = "deadline_minus_45m_reached_after_completed_fold"
            break
    status["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    status["remaining_seconds"] = remaining_seconds()
    atomic_json(OUT / "audit/fix2_supervisor_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
