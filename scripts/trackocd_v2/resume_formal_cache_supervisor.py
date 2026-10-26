#!/usr/bin/env python3
"""Resume one bounded formal COVTrack feature-cache range safely.

This supervisor owns only the cache worker that it starts.  It waits for the
worker's actual resource contract (25% of RAM plus the declared 8 GiB worker
margin, and an idle GPU 0), terminates only that worker after repeated drops
below the hard RAM floor, and resumes at shard boundaries.  It never selects a
different frontend, opens Test, or touches unrelated processes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.trackocd_v2.build_common_features import (  # noqa: E402
    WORKER_RAM_GIB,
    resource_snapshot,
)
from src.trackocd_v2.io import atomic_json  # noqa: E402


DEFAULT_CACHE_ROOT = ROOT / "outputs/trackocd_v2_runtime/features/formal/covtrack_native_continuation_20261022"
DEFAULT_INPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/nas_recovery_20261022/val/covtrack_native_physical_tracks.jsonl"
DEFAULT_MODEL_REPO = ROOT / "vendor/trackocd_core/dinov2_7764ea0"
DEFAULT_MODEL_WEIGHTS = ROOT / "checkpoints/dinov2_vitb14_pretrain.pth"
DEFAULT_AUDIT_ROOT = ROOT / "outputs/trackocd_cvpr/cvpr2/nas_recovery_20261022/audit"
DEFAULT_FRAMES_ROOT = Path("/data3/liuyeqiang/TAO-Amodal/frames")
DEFAULT_RECEIPT = ROOT / "outputs/trackocd_cvpr/cvpr5/formal_covtrack_cache_resume_supervisor_20261026.json"
DEFAULT_LOG = ROOT / "outputs/trackocd_cvpr/cvpr5/formal_covtrack_cache_resume_supervisor_20261026.log"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def done_count(cache_root: Path, start: int, end: int) -> int:
    shards = cache_root / "shards"
    return sum(
        (shards / (f"shard-{index:06d}") / ".done").is_file()
        for index in range(start, end)
    )


def gate(snapshot: dict[str, Any], requested_gpu_index: int) -> dict[str, Any]:
    total = int(snapshot.get("mem_total_kib") or 0)
    available = int(snapshot.get("mem_available_kib") or 0)
    ram_floor = int(total * 0.25)
    worker_margin = int(WORKER_RAM_GIB * 1024 * 1024)
    required = ram_floor + worker_margin
    candidates = sorted(snapshot.get("gpu_rows", []), key=lambda row: int(row.get("index", -1)))
    if requested_gpu_index >= 0:
        candidates = [row for row in candidates if int(row.get("index", -1)) == requested_gpu_index]
    gpu = next(
        (row for row in candidates if int(row.get("memory_free_mib", 0)) >= 8192 and not row.get("compute_apps")),
        None,
    )
    selected_gpu_index = int(gpu["index"]) if gpu is not None else None
    gpu_ok = gpu is not None
    return {
        "mem_total_kib": total,
        "mem_available_kib": available,
        "ram_floor_kib": ram_floor,
        "worker_margin_kib": worker_margin,
        "required_available_kib": required,
        "ram_floor_pass": available >= ram_floor,
        "worker_gate_pass": available >= required,
        "requested_gpu_index": requested_gpu_index,
        "gpu_index": selected_gpu_index,
        "gpu_gate_pass": gpu_ok,
        "gpu": gpu,
        "pass": available >= required and gpu_ok,
    }


def resource_failure(cache_root: Path, start: int, end: int) -> dict[str, Any] | None:
    path = cache_root / f"cache_failure.worker-{start:06d}-{end:06d}.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": str(path), "error": "unreadable failure receipt"}
    value["path"] = str(path)
    return value


def stop_owned_child(process: subprocess.Popen[Any]) -> str:
    process.terminate()
    try:
        process.wait(timeout=20)
        return "SIGTERM"
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)
        return "SIGTERM_THEN_SIGKILL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-repo", type=Path, default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-weights", type=Path, default=DEFAULT_MODEL_WEIGHTS)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--start", type=int, default=399)
    parser.add_argument("--end", type=int, default=723)
    parser.add_argument("--gpu-index", type=int, default=-1, help="GPU index, or -1 to choose a currently eligible idle GPU")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    if args.start < 0 or args.end <= args.start:
        raise SystemExit("invalid half-open shard range")
    if args.batch < 1 or args.batch > 64:
        raise SystemExit("--batch must be between 1 and 64")

    args.cache_root = args.cache_root.resolve()
    args.input = args.input.resolve()
    args.model_repo = args.model_repo.resolve()
    args.model_weights = args.model_weights.resolve()
    args.audit_root = args.audit_root.resolve()
    args.frames_root = args.frames_root.resolve()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    def build_command(selected_gpu_index: int) -> list[str]:
        return [
            sys.executable,
            str(ROOT / "scripts/trackocd_v2/build_sharded_feature_cache.py"),
            "--frontend", "covtrack_native",
            "--split", "val",
            "--input", str(args.input),
            "--output-root", str(args.cache_root),
            "--model-repo", str(args.model_repo),
            "--model-weights", str(args.model_weights),
            "--audit-root", str(args.audit_root),
            "--frames-root", str(args.frames_root),
            "--batch", str(args.batch),
            "--shard-tracks", "512",
            "--gpu-index", str(selected_gpu_index),
            "--shard-start", str(args.start),
            "--shard-end", str(args.end),
        ]

    started = utc_now()
    prior_failure = resource_failure(args.cache_root, args.start, args.end)
    events: list[dict[str, Any]] = []
    process: subprocess.Popen[Any] | None = None
    below_floor_samples = 0
    status = "WAITING_RESOURCE"
    child_pid: int | None = None
    command: list[str] | None = None

    def write_receipt(latest: dict[str, Any]) -> None:
        payload = {
            "schema_version": "trackocd.cvpr5.formal_covtrack_cache_resume_supervisor.v1",
            "status": status,
            "generated_utc": utc_now(),
            "started_utc": started,
            "frontend": "COVTrack-native",
            "split": "val",
            "cache_root": str(args.cache_root),
            "input": str(args.input),
            "range": {"start_inclusive": args.start, "end_exclusive": args.end},
            "command": command,
            "batch": args.batch,
            "gpu_index": args.gpu_index,
            "done_count": done_count(args.cache_root, args.start, args.end),
            "prior_failure": prior_failure,
            "latest": latest,
            "events": events[-100:],
            "safety": {
                "tao_test_read": False,
                "external_processes_touched": False,
                "only_owned_child_terminated": True,
            },
        }
        atomic_json(args.receipt, payload)

    def announce(payload: dict[str, Any]) -> None:
        print(json.dumps(payload, sort_keys=True), flush=True)

    try:
        while True:
            snapshot = resource_snapshot()
            resource_gate = gate(snapshot, args.gpu_index)
            current_done = done_count(args.cache_root, args.start, args.end)

            if process is None:
                if current_done == args.end - args.start:
                    status = "RANGE_COMPLETE"
                    event = {"time": utc_now(), "event": status, "done_count": current_done}
                    events.append(event)
                    write_receipt({"event": status, "resource_gate": resource_gate})
                    announce({"supervisor": status, "done": current_done})
                    return 0
                if not resource_gate["pass"]:
                    status = "WAITING_RESOURCE"
                    latest = {
                        "event": status,
                        "done_count": current_done,
                        "resource_gate": resource_gate,
                    }
                    write_receipt(latest)
                    announce({"supervisor": status, "done": current_done, "mem_available_kib": resource_gate["mem_available_kib"], "required_available_kib": resource_gate["required_available_kib"]})
                    time.sleep(max(1.0, args.poll_seconds))
                    continue

                status = "CHILD_RUNNING"
                selected_gpu_index = resource_gate["gpu_index"]
                if selected_gpu_index is None:
                    raise RuntimeError("resource gate passed without selecting a GPU")
                command = build_command(int(selected_gpu_index))
                log_handle = args.log.open("a", encoding="utf-8")
                log_handle.write(f"\n[{utc_now()}] launch: {json.dumps(command)}\n")
                log_handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                log_handle.close()
                child_pid = process.pid
                below_floor_samples = 0
                events.append({"time": utc_now(), "event": "CHILD_STARTED", "child_pid": child_pid, "done_before": current_done, "resource_gate": resource_gate})
                write_receipt({"event": "CHILD_STARTED", "child_pid": child_pid, "resource_gate": resource_gate})
                announce({"supervisor": "CHILD_STARTED", "child_pid": child_pid, "done_before": current_done})
                continue

            return_code = process.poll()
            current_done = done_count(args.cache_root, args.start, args.end)
            if return_code is not None:
                failure = resource_failure(args.cache_root, args.start, args.end)
                resource_retry = bool(failure and "WAITING_RESOURCE" in str(failure.get("error", "")))
                if return_code == 0 and current_done == args.end - args.start:
                    status = "RANGE_COMPLETE"
                    events.append({"time": utc_now(), "event": "CHILD_EXIT", "child_pid": child_pid, "returncode": return_code, "done_count": current_done})
                    write_receipt({"event": "CHILD_EXIT", "returncode": return_code, "done_count": current_done})
                    announce({"supervisor": "CHILD_EXIT", "returncode": return_code, "done": current_done})
                    return 0
                if return_code == 2 or resource_retry:
                    status = "WAITING_RESOURCE"
                    events.append({"time": utc_now(), "event": "RESOURCE_RETRY", "child_pid": child_pid, "returncode": return_code, "done_count": current_done, "failure": failure})
                    write_receipt({"event": "RESOURCE_RETRY", "returncode": return_code, "done_count": current_done, "failure": failure})
                    announce({"supervisor": "RESOURCE_RETRY", "returncode": return_code, "done": current_done})
                    process = None
                    child_pid = None
                    time.sleep(max(1.0, args.poll_seconds))
                    continue
                status = "FAILED_NONRESOURCE"
                events.append({"time": utc_now(), "event": status, "child_pid": child_pid, "returncode": return_code, "done_count": current_done, "failure": failure})
                write_receipt({"event": status, "returncode": return_code, "done_count": current_done, "failure": failure})
                announce({"supervisor": status, "returncode": return_code, "done": current_done})
                return 1

            if not resource_gate["ram_floor_pass"]:
                below_floor_samples += 1
            else:
                below_floor_samples = 0
            if below_floor_samples >= 2:
                stop_mode = stop_owned_child(process)
                events.append({"time": utc_now(), "event": "CHILD_STOPPED_FOR_RAM", "child_pid": child_pid, "stop_mode": stop_mode, "done_count": current_done, "resource_gate": resource_gate})
                status = "WAITING_RESOURCE"
                write_receipt({"event": "CHILD_STOPPED_FOR_RAM", "child_pid": child_pid, "done_count": current_done, "resource_gate": resource_gate})
                announce({"supervisor": "CHILD_STOPPED_FOR_RAM", "child_pid": child_pid, "done": current_done})
                process = None
                child_pid = None
                below_floor_samples = 0
                time.sleep(max(1.0, args.poll_seconds))
                continue

            write_receipt({"event": "CHILD_RUNNING", "child_pid": child_pid, "done_count": current_done, "resource_gate": resource_gate})
            announce({"supervisor": "CHILD_RUNNING", "child_pid": child_pid, "done": current_done, "mem_available_kib": resource_gate["mem_available_kib"]})
            time.sleep(max(1.0, args.poll_seconds))
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            stop_mode = stop_owned_child(process)
            events.append({"time": utc_now(), "event": "INTERRUPTED", "child_pid": child_pid, "stop_mode": stop_mode})
        status = "INTERRUPTED"
        write_receipt({"event": status, "child_pid": child_pid})
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
