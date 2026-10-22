#!/usr/bin/env python3
"""Wait for the formal Val feature cache, then run Val-only CPU baselines.

This is a task-owned continuation gate. It never launches a cache worker, never
opens Test assets, and only starts the evaluator after the complete cache has
passed the reusable numeric payload audit. A single process runs nearest and
DP-means sequentially, preserving any sealed prefix outputs on retry.
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
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_v2_runtime"
DEFAULT_AUDIT = ROOT / "outputs/trackocd_cvpr/cvpr2/nas_recovery_20261022/audit"
DEFAULT_STREAM = ROOT / "outputs/trackocd_cvpr/cvpr2/nas_recovery_20261022/val/covtrack_native_physical_tracks.jsonl"
DEFAULT_CACHE_MANIFEST = DEFAULT_OUTPUT / "features/formal/covtrack_native_continuation_20261022/cache_manifest.json"
DEFAULT_LOG = ROOT / "outputs/trackocd_cvpr/cvpr2/formal_baseline_gate_20261022.log"
SOURCE_SHA256 = "63945d2ae19f63390930b3ad0ed1971d64c4d3ff1067fff2842317f31d8fe2f1"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _mem_available() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        fields = rest.split()
        if fields:
            values[key] = int(fields[0])
    return int(values.get("MemAvailable", 0)), int(values.get("MemTotal", 0))


def _cache_state(manifest: Path) -> tuple[str, dict[str, Any]]:
    if not manifest.is_file():
        return "MISSING", {}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        total_tracks = int(payload.get("total_tracks", 0))
        shard_tracks = int(payload.get("shard_tracks", 0))
        required = (total_tracks + shard_tracks - 1) // shard_tracks if shard_tracks else 0
        completed = len(payload.get("completed_shards") or [])
        if payload.get("status") == "COMPLETE" and required and completed == required:
            return "COMPLETE", payload
        return f"{payload.get('status', 'UNKNOWN')}:{completed}/{required}", payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return f"INVALID:{type(exc).__name__}", {}


def _event(handle, event: str, **fields: Any) -> None:
    payload = {"event": event, "time": _now(), **fields}
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()
    print(json.dumps(payload, sort_keys=True), flush=True)


def _run_audit(manifest: Path, output: Path, log_handle) -> dict[str, Any]:
    audit_output = output / "audit/formal_cache_payload_audit_20261022.json"
    command = [
        sys.executable,
        str((ROOT / "scripts/trackocd_v2/audit_sharded_cache_payload.py").resolve()),
        "--manifest", str(manifest.resolve()),
        "--expected-source-sha256", SOURCE_SHA256,
        "--output", str(audit_output.resolve()),
    ]
    result = subprocess.run(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"payload audit failed with return code {result.returncode}")
    payload = json.loads(audit_output.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS_COMPLETE" or payload.get("formal_cache_complete") is not True:
        raise RuntimeError(f"payload audit did not certify complete cache: {payload.get('status')}")
    if int(payload.get("error_count", -1)) != 0 or int(payload.get("invalid_box_count", -1)) != 0:
        raise RuntimeError("complete-cache payload audit has errors or invalid boxes")
    return payload


def _run_baseline(method: str, output: Path, audit: Path, stream: Path, manifest: Path, log_handle) -> None:
    table = output / "tables" / f"pred_{method}.json"
    if table.is_file():
        try:
            existing = json.loads(table.read_text(encoding="utf-8"))
            if existing.get("status") == "COMPLETE":
                _event(log_handle, "BASELINE_ALREADY_COMPLETE", method=method, table=str(table.resolve()))
                return
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    command = [
        sys.executable,
        str((ROOT / "scripts/trackocd_v2/run_pred_baselines.py").resolve()),
        "--method", method,
        "--device", "cpu",
        "--output-root", str(output.resolve()),
        "--audit-root", str(audit.resolve()),
        "--stream", str(stream.resolve()),
        "--cache-manifest", str(manifest.resolve()),
    ]
    _event(log_handle, "BASELINE_START", method=method, command=command)
    result = subprocess.run(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
    _event(log_handle, "BASELINE_EXIT", method=method, returncode=result.returncode)
    if result.returncode != 0:
        raise RuntimeError(f"{method} baseline failed with return code {result.returncode}")


def run(output: Path, audit: Path, stream: Path, manifest: Path, log: Path, sleep_seconds: int) -> int:
    output.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    last_state = None
    with log.open("a", encoding="utf-8") as handle:
        _event(handle, "GATE_START", manifest=str(manifest.resolve()), test_semantic_accessed=False)
        while True:
            state, payload = _cache_state(manifest)
            available, total = _mem_available()
            if state != last_state:
                _event(handle, "CACHE_STATE", state=state, mem_available_kib=available, mem_total_kib=total)
                last_state = state
            if state == "COMPLETE":
                if total and available < total // 4:
                    time.sleep(sleep_seconds)
                    continue
                _event(handle, "CACHE_COMPLETE", completed_shards=len(payload.get("completed_shards") or []))
                _run_audit(manifest, output, handle)
                _event(handle, "PAYLOAD_AUDIT_PASS", test_semantic_accessed=False)
                for method in ("nearest", "dpmeans"):
                    _run_baseline(method, output, audit, stream, manifest, handle)
                _event(handle, "BASELINES_COMPLETE", methods=["nearest", "dpmeans"], test_semantic_accessed=False)
                return 0
            time.sleep(sleep_seconds)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--sleep-seconds", type=int, default=60)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.sleep_seconds < 1:
        raise ValueError("sleep-seconds must be positive")
    return run(args.output_root.resolve(), args.audit_root.resolve(), args.stream.resolve(), args.cache_manifest.resolve(), args.log.resolve(), args.sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
