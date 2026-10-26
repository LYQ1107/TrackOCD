#!/usr/bin/env python3
"""Finalize the complete Val cache, then run the sealed CPU Val baselines.

The process only waits while the GPU cache supervisor owns extraction.  After
all expected atomic shard markers exist and no shard is still marked launched,
it regenerates the non-range manifest, runs the independent payload audit, and
delegates to the existing Val-only nearest/DP-means gate.  It never opens Test
assets, starts a GPU worker, or terminates another process.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import atomic_json  # noqa: E402


DEFAULT_CACHE_ROOT = ROOT / "outputs/trackocd_v2_runtime/features/formal/covtrack_native_continuation_20261022"
DEFAULT_INPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/nas_recovery_20261022/val/covtrack_native_physical_tracks.jsonl"
DEFAULT_MODEL_REPO = ROOT / "vendor/trackocd_core/dinov2_7764ea0"
DEFAULT_MODEL_WEIGHTS = ROOT / "checkpoints/dinov2_vitb14_pretrain.pth"
DEFAULT_AUDIT_ROOT = ROOT / "outputs/trackocd_cvpr/cvpr2/nas_recovery_20261022/audit"
DEFAULT_FRAMES_ROOT = Path("/data3/liuyeqiang/TAO-Amodal/frames")
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/trackocd_v2_runtime"
DEFAULT_RECEIPT = ROOT / "outputs/trackocd_cvpr/cvpr5/formal_cache_baseline_postprocess_20261026.json"
DEFAULT_LOG = ROOT / "outputs/trackocd_cvpr/cvpr5/formal_cache_baseline_postprocess_20261026.log"
EXPECTED_SHARDS = 723


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def memory_snapshot() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        fields = rest.split()
        if fields:
            values[key] = int(fields[0])
    return int(values.get("MemAvailable", 0)), int(values.get("MemTotal", 0))


def done_and_launched(cache_root: Path, expected: int) -> tuple[int, int]:
    shards = cache_root / "shards"
    done = 0
    launched = 0
    for index in range(expected):
        shard = shards / f"shard-{index:06d}"
        done += int((shard / ".done").is_file())
        launched += int((shard / ".launched").is_file())
    return done, launched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-repo", type=Path, default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-weights", type=Path, default=DEFAULT_MODEL_WEIGHTS)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        raise SystemExit("--poll-seconds must be positive")

    for name in ("cache_root", "input", "model_repo", "model_weights", "audit_root", "frames_root", "output_root", "receipt", "log"):
        setattr(args, name, getattr(args, name).resolve())
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    manifest = args.cache_root / "cache_manifest.json"
    builder_command = [
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
        "--batch", "16",
        "--shard-tracks", "512",
    ]
    baseline_command = [
        sys.executable,
        str(ROOT / "scripts/trackocd_v2/run_formal_baselines_after_cache.py"),
        "--output-root", str(args.output_root),
        "--audit-root", str(args.audit_root),
        "--stream", str(args.input),
        "--cache-manifest", str(manifest),
        "--log", str(args.log),
        "--sleep-seconds", "60",
    ]

    status = "WAITING_CACHE"
    events: list[dict[str, Any]] = []
    started = utc_now()

    def write_receipt(latest: dict[str, Any]) -> None:
        available, total = memory_snapshot()
        done, launched = done_and_launched(args.cache_root, EXPECTED_SHARDS)
        payload = {
            "schema_version": "trackocd.cvpr5.formal_cache_baseline_postprocess.v1",
            "status": status,
            "generated_utc": utc_now(),
            "started_utc": started,
            "frontend": "COVTrack-native",
            "split": "val",
            "cache_root": str(args.cache_root),
            "manifest": str(manifest),
            "input": str(args.input),
            "expected_shards": EXPECTED_SHARDS,
            "done_shards": done,
            "launched_markers": launched,
            "mem_available_kib": available,
            "mem_total_kib": total,
            "builder_command": builder_command,
            "baseline_command": baseline_command,
            "latest": latest,
            "events": events[-100:],
            "safety": {
                "tao_test_read": False,
                "gpu_worker_started": False,
                "external_processes_touched": False,
            },
        }
        atomic_json(args.receipt, payload)

    def emit(handle, event: str, **fields: Any) -> None:
        payload = {"event": event, "time": utc_now(), **fields}
        events.append(payload)
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        print(json.dumps(payload, sort_keys=True), flush=True)

    last_wait_state: tuple[int, int, int] | None = None
    with args.log.open("a", encoding="utf-8") as handle:
        while True:
            done, launched = done_and_launched(args.cache_root, EXPECTED_SHARDS)
            available, total = memory_snapshot()
            wait_state = (done, launched, available // (1024 * 1024))
            if (done, launched) != (EXPECTED_SHARDS, 0):
                status = "WAITING_CACHE"
                if wait_state != last_wait_state:
                    emit(handle, status, done_shards=done, launched_markers=launched, mem_available_kib=available)
                    write_receipt({"event": status, "done_shards": done, "launched_markers": launched})
                    last_wait_state = wait_state
                time.sleep(args.poll_seconds)
                continue

            if total and available < total // 4:
                status = "WAITING_RAM"
                emit(handle, status, done_shards=done, mem_available_kib=available, ram_floor_kib=total // 4)
                write_receipt({"event": status, "done_shards": done, "mem_available_kib": available})
                time.sleep(args.poll_seconds)
                continue

            status = "FINALIZING_MANIFEST"
            emit(handle, status, command=builder_command)
            result = subprocess.run(builder_command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                status = "FAILED_MANIFEST_FINALIZATION"
                emit(handle, status, returncode=result.returncode)
                write_receipt({"event": status, "returncode": result.returncode})
                return result.returncode
            try:
                manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
                complete = manifest_payload.get("status") == "COMPLETE" and len(manifest_payload.get("completed_shards") or []) == EXPECTED_SHARDS
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                complete = False
            if not complete:
                status = "FAILED_MANIFEST_FINALIZATION"
                emit(handle, status, reason="non-range manifest is not COMPLETE")
                write_receipt({"event": status, "reason": "non-range manifest is not COMPLETE"})
                return 1

            status = "RUNNING_VAL_BASELINE_GATE"
            emit(handle, status, command=baseline_command)
            result = subprocess.run(baseline_command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                status = "FAILED_VAL_BASELINE_GATE"
                emit(handle, status, returncode=result.returncode)
                write_receipt({"event": status, "returncode": result.returncode})
                return result.returncode
            status = "COMPLETE_VAL_BASELINES"
            emit(handle, status, metrics_scope="Val-only CPU nearest and DP-means", test_semantic_accessed=False)
            write_receipt({"event": status, "done_shards": EXPECTED_SHARDS})
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
