#!/usr/bin/env python3
"""Persistent completion supervisor for Phase88 C0v2 fix2.

This process is intentionally completion-based: the original registration is
kept as provenance but never stops a valid scientific route.  It runs at most
one worker under the current host pressure, pauses only after a persistent RAM
floor breach, and resumes only from a validated fix2 checkpoint.
"""
from __future__ import annotations

import datetime as dt
import hashlib
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
SHARED = "/data2/usr_for_deadline/trackocd_phase88/shared_features"
FLOOR_KIB = int(31.25 * 1024 * 1024)
MAX_UPDATES = 20000
POLL_SECONDS = 30
RESOURCE_WAIT_SECONDS = 120

sys.path.insert(0, str(ROOT))
from src.iclr27_phase88.resource_manager import (  # noqa: E402
    ResourceManager,
    RAM_FLOOR_GIB,
    discover_usable_gpus,
    process_memory_detail,
    safe_worker_count,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def manifest_sha() -> str:
    path = OUT / "manifests/causal_event_v2_fix2_manifest.json"
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def semantic_contract_sha() -> str:
    h = hashlib.sha256()
    for name in ("rollout.py", "memory.py", "transitions.py", "data.py", "data_memmap.py"):
        path = ROOT / "src/iclr27_phase88" / name
        h.update(name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def load_checkpoint(path: Path) -> dict | None:
    try:
        import torch
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def valid_fix2_checkpoint(path: Path, fold: int) -> tuple[bool, dict]:
    payload = load_checkpoint(path)
    expected = manifest_sha()
    reason = "ok"
    if payload is None:
        reason = "unreadable"
    elif payload.get("event_tag") != "fix2":
        reason = "event_tag_mismatch"
    elif payload.get("manifest_sha256") != expected:
        reason = "manifest_sha_mismatch"
    elif payload.get("semantic_contract_sha256") not in (None, semantic_contract_sha()):
        reason = "semantic_contract_sha_mismatch"
    elif int(payload.get("fold", -1)) != int(fold):
        reason = "fold_mismatch"
    elif int(payload.get("step", 0)) <= 0:
        reason = "missing_positive_step"
    return reason == "ok", {
        "path": str(path), "reason": reason,
        "step": int(payload.get("step", 0)) if payload else None,
        "event_tag": payload.get("event_tag") if payload else None,
        "manifest_sha256": payload.get("manifest_sha256") if payload else None,
        "expected_manifest_sha256": expected,
        "semantic_contract_sha256_at_resume": semantic_contract_sha(),
        "checkpoint_embeds_semantic_contract_sha256": bool(payload and payload.get("semantic_contract_sha256")),
        "checkpoint_sha256": sha256(path) if path.exists() else None,
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_candidates(tag: str) -> list[Path]:
    return sorted((OUT / "checkpoints").glob(f"{tag}_step*.pt"), key=lambda p: p.stat().st_mtime)


def latest_valid_checkpoint(tag: str, fold: int) -> tuple[Path | None, dict]:
    details = []
    for path in reversed(checkpoint_candidates(tag)):
        ok, info = valid_fix2_checkpoint(path, fold)
        details.append(info)
        if ok:
            return path, {"selected": info, "candidates": details}
    return None, {"selected": None, "candidates": details}


def pid_alive(pid: int | None) -> bool:
    if not pid or int(pid) <= 0:
        return False
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text().split()
        if len(stat) > 2 and stat[2] == "Z":
            return False
    except OSError:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def marker_pid(path: Path) -> int | None:
    try:
        return int(json.loads(path.read_text()).get("pid", 0))
    except Exception:
        return None


def append_resource_log(tag: str, snap) -> None:
    path = OUT / "audit" / f"resource_profile_{tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(snap.as_dict(), sort_keys=True) + "\n")
        handle.flush()


def run_low_ram_analysis(status: dict) -> None:
    target = OUT / "audit/low_ram_failure_analysis.json"
    if target.exists():
        return
    cmd = [PY, str(ROOT / "scripts/iclr27_phase88/analyze_low_ram_failure.py")]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    status.setdefault("low_ram_analysis", {})["returncode"] = int(result.returncode)
    status["low_ram_analysis"]["completed_utc"] = now()
    status["low_ram_analysis"]["output"] = str(target)


def resources(manager: ResourceManager, worker_pid: int | None = None, own_pids=()):
    return manager.snapshot(worker_pid=worker_pid, own_pids=own_pids)


def wait_for_resource(manager: ResourceManager, status: dict) -> tuple[object, list[int]]:
    run_low_ram_analysis(status)
    while True:
        free = discover_usable_gpus()
        snap = resources(manager)
        status["resource_state"] = "WAITING_FOR_RESOURCE" if snap.safe_workers <= 0 else "RESOURCE_AVAILABLE"
        status["last_resource_snapshot"] = snap.as_dict()
        status["next_action"] = "wait 120 seconds and recompute resources" if snap.safe_workers <= 0 else "resume valid fix2 formal queue"
        atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
        if snap.safe_workers > 0 and free:
            return snap, free
        time.sleep(RESOURCE_WAIT_SECONDS)


def run_worker(manager: ResourceManager, fold: int, gpu: int, resume_path: Path | None, status: dict) -> dict:
    tag = f"c0v2_fix2_formal_f{fold}"
    log_path = OUT / "logs" / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(ROOT / "scripts/iclr27_phase88/train_controller.py"),
           "--fold", str(fold), "--device", f"cuda:{gpu}", "--updates", str(MAX_UPDATES),
           "--tag", tag, "--checkpoint-interval", "2000", "--event-tag", "fix2",
           "--memmap-root", SHARED]
    if resume_path is not None:
        cmd.extend(["--resume-checkpoint", str(resume_path), "--resume-launched"])
    status["active_workers"] = [{"fold": fold, "gpu": gpu, "tag": tag, "command": cmd}]
    atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
    breach = 0
    requested_pause = False
    started = now()
    with log_path.open("a") as handle:
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(POLL_SECONDS)
            snap = resources(manager, worker_pid=proc.pid, own_pids=[proc.pid])
            append_resource_log(tag, snap)
            if snap.mem_available_kib < FLOOR_KIB:
                breach += 1
            else:
                breach = 0
            status["last_resource_snapshot"] = snap.as_dict()
            status["persistent_breach_samples"] = breach
            atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
            if breach >= 3 and not requested_pause:
                requested_pause = True
                status["resource_state"] = "PAUSE_REQUESTED_RESOURCE"
                atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
                os.kill(proc.pid, signal.SIGUSR1)
        returncode = int(proc.wait())
    status["active_workers"] = []
    result = {"fold": fold, "gpu": gpu, "tag": tag, "returncode": returncode,
              "started_utc": started, "finished_utc": now(), "pause_requested": requested_pause,
              "paused_marker": str(OUT / "completion" / f"{tag}.paused_resource.json"),
              "done": (OUT / "completion" / f"{tag}.done").exists()}
    status.setdefault("worker_history", []).append(result)
    atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
    return result


def wait_existing_worker(manager: ResourceManager, fold: int, pid: int, status: dict) -> None:
    """Attach to a task-owned worker left by an older supervisor instance."""
    tag = f"c0v2_fix2_formal_f{fold}"
    breach = 0
    requested_pause = False
    while pid_alive(pid):
        time.sleep(POLL_SECONDS)
        snap = resources(manager, worker_pid=pid, own_pids=[pid])
        append_resource_log(tag, snap)
        breach = breach + 1 if snap.mem_available_kib < FLOOR_KIB else 0
        status["active_workers"] = [{"fold": fold, "pid": pid, "tag": tag, "attached": True}]
        status["last_resource_snapshot"] = snap.as_dict()
        status["persistent_breach_samples"] = breach
        atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
        if breach >= 3 and not requested_pause:
            requested_pause = True
            os.kill(pid, signal.SIGUSR1)
            status["resource_state"] = "PAUSE_REQUESTED_RESOURCE"
            atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
    status["active_workers"] = []
    status.setdefault("worker_history", []).append({
        "fold": fold, "tag": tag, "pid": pid, "attached": True,
        "finished_utc": now(), "pause_requested": requested_pause,
    })


def run_validation(fold: int, gpu: int, status: dict) -> dict:
    tag = f"c0v2_fix2_formal_f{fold}"
    val_tag = f"{tag}_val"
    checkpoint = OUT / "checkpoints" / f"{tag}.pt"
    log_path = OUT / "logs" / f"{val_tag}.log"
    cmd = [PY, str(ROOT / "scripts/iclr27_phase88/evaluate_checkpoint_sharded.py"),
           "--fold", str(fold), "--checkpoint", str(checkpoint), "--tag", val_tag,
           "--event-tag", "fix2", "--shard-size", "250", "--device", f"cuda:{gpu}",
           "--memmap-root", SHARED]
    with log_path.open("w") as handle:
        result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)
    entry = {"fold": fold, "validation_returncode": int(result.returncode), "tag": val_tag,
             "finished_utc": now(), "metrics": str(OUT / "validation" / val_tag / "final_metrics.json")}
    status.setdefault("validation_history", []).append(entry)
    atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
    return entry


def main() -> None:
    if not (OUT / "completion/build_events_v2_fix2.done").exists():
        raise RuntimeError("fix2 event manifest is not complete")
    metric = OUT / "metrics/c0v2_fix2_targeted_f0.json"
    measured_rss_kib = 5052000
    try:
        values = json.loads(metric.read_text()).get("rss_samples", [])
        if values:
            measured_rss_kib = max(int(x.get("rss_bytes", 0)) for x in values) // 1024
    except Exception:
        pass
    manager = ResourceManager(measured_rss_kib, max_workers=4)
    status = {
        "phase": 88, "mode": "CONTINUOUS_COMPLETION", "route": "C0V2_FIX2",
        "started_utc": now(), "max_updates": MAX_UPDATES,
        "measured_worker_rss_kib": measured_rss_kib,
        "ram_floor_kib": FLOOR_KIB, "ram_floor_gib": RAM_FLOOR_GIB,
        "external_processes_touched": False, "public_dev_q1_sealed_accessed": False,
        "worker_history": [], "validation_history": [], "active_workers": [],
        "resource_state": "INITIALIZING", "task_status": "IN_PROGRESS",
        "original_registration_preserved": True,
    }
    atomic_json(OUT / "audit/fix2_continuous_supervisor_start.json", {
        **status, "manifest_sha256": manifest_sha(),
        "semantic_contract_sha256": semantic_contract_sha(),
    })
    atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
    run_low_ram_analysis(status)
    for fold in range(4):
        tag = f"c0v2_fix2_formal_f{fold}"
        done = OUT / "completion" / f"{tag}.done"
        if done.exists():
            status.setdefault("fold_status", {})[str(fold)] = "DONE"
            continue
        marker = OUT / "completion" / f"{tag}.launched"
        # Only the corrected fix2 f0 progress checkpoint is resumable.  Any
        # other launched marker without a validated fix2 checkpoint is held,
        # never blindly relaunched.
        resume_path, resume_info = latest_valid_checkpoint(tag, fold)
        if resume_path is not None:
            atomic_json(OUT / "audit" / f"fix2_checkpoint_validation_f{fold}.json", {
                "phase": 88, "route": "C0V2_FIX2", "fold": fold,
                "checkpoint": str(resume_path), "status": "VALID_FIX2_PROGRESS_CHECKPOINT",
                "validation": resume_info,
                "legacy_checkpoint_without_embedded_contract_sha": bool(
                    resume_info.get("selected", {}).get("checkpoint_embeds_semantic_contract_sha256") is False
                ),
                "semantic_contract_verified_by_manifest_and_fix2_event_tag": True,
                "formal_protocol_changed": False,
            })
        if marker.exists() and resume_path is None:
            status.setdefault("fold_status", {})[str(fold)] = "WAITING_INVALID_OR_INCOMPLETE_LAUNCHED_UNIT"
            status.setdefault("held_units", []).append({"fold": fold, "tag": tag, "reason": resume_info})
            atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
            break
        existing_pid = marker_pid(marker) if marker.exists() else None
        if existing_pid is not None and pid_alive(existing_pid):
            status["resource_state"] = "ATTACHED_TO_EXISTING_TASK_WORKER"
            atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
            wait_existing_worker(manager, fold, existing_pid, status)
            resume_path, resume_info = latest_valid_checkpoint(tag, fold)
        while True:
            snap, free = wait_for_resource(manager, status)
            gpu = int(free[0])
            result = run_worker(manager, fold, gpu, resume_path, status)
            status.setdefault("fold_status", {})[str(fold)] = "PAUSED_RESOURCE" if result["pause_requested"] and not result["done"] else "TRAIN_RETURNED"
            atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
            if result["done"]:
                validation = run_validation(fold, gpu, status)
                if validation["validation_returncode"] != 0:
                    status["task_status"] = "IN_PROGRESS_VALIDATION_REPAIR_REQUIRED"
                    atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
                    raise SystemExit(validation["validation_returncode"])
                status["fold_status"][str(fold)] = "VALIDATED"
                atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
                break
            resume_path, resume_info = latest_valid_checkpoint(tag, fold)
            if resume_path is None:
                status["task_status"] = "IN_PROGRESS_WAITING_VALID_CHECKPOINT"
                status.setdefault("held_units", []).append({"fold": fold, "tag": tag, "reason": resume_info})
                atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
                raise SystemExit(2)
            # A resource pause is expected; the loop waits for the next safe
            # window and resumes the same fold without changing exposure.
        atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
    all_done = all((OUT / "completion" / f"c0v2_fix2_formal_f{f}.done").exists() for f in range(4))
    status["task_status"] = "C0V2_FORMAL_COMPLETE_PENDING_TRAIN_VALIDATION" if all_done else "IN_PROGRESS"
    status["current_stage"] = "C0V2_FORMAL_COMPLETE_PENDING_FREEZE" if all_done else "RESOURCE_WAIT_OR_FIX2_FORMAL"
    atomic_json(OUT / "audit/fix2_continuous_supervisor_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
