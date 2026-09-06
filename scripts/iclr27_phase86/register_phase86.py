#!/usr/bin/env python3
"""Register the Phase86 window and immutable resource/sealing boundary."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase86" / "audit"


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()
    except Exception as exc:  # pragma: no cover - audit best effort
        return f"ERROR: {type(exc).__name__}: {exc}"


def sha(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    deadline = now + dt.timedelta(hours=10)
    out = {
        "schema_version": "trackocd.phase86.window_registration.v1",
        "phase": 86,
        "project_root": str(ROOT),
        "luna_session_id": "01a01fb6-96f7-7132-a318-0833180c88d8",
        "host": "remote-ssh-discovered:gpu80-user",
        "start_utc": now.isoformat().replace("+00:00", "Z"),
        "deadline_utc": deadline.isoformat().replace("+00:00", "Z"),
        "start_head": run(["git", "rev-parse", "HEAD"]),
        "origin_main_head": run(["git", "ls-remote", "origin", "refs/heads/main"]).split()[0]
        if run(["git", "ls-remote", "origin", "refs/heads/main"]) else None,
        "git_status": run(["git", "status", "--short", "--branch"]),
        "git_log": run(["git", "log", "-30", "--oneline"]),
        "date_utc": run(["date", "-u"]),
        "nvidia_smi": run(["nvidia-smi"]),
        "free_h": run(["free", "-h"]),
        "disk_data1": run(["df", "-h", "/data1"]),
        "disk_data2": run(["df", "-h", "/data2"]),
        "process_count": len(run(["ps", "-eo", "pid="]).splitlines()),
        "phase85_read_only": True,
        "public_dev_q1_sealed_accessed": False,
        "diagnostic_only": True,
        "max_gpus": 4,
        "gpu_policy": "use only idle task-owned GPUs; never touch external PID 33785 on GPU0",
        "large_output_root": "/data2/usr_for_deadline/trackocd_phase86/project_outputs",
        "phase85_report_sha256": sha(ROOT / "docs/iclr27_phase85/PHASE85_AUTONOMOUS_RESEARCH_REPORT.md"),
        "phase85_decision_sha256": sha(ROOT / "outputs/iclr27_phase85/audit/phase85_decision.json"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / ".window_registration.json.tmp"
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(OUT / "window_registration.json")
    print(json.dumps({k: out[k] for k in ("start_utc", "deadline_utc", "start_head", "origin_main_head")}, indent=2))


if __name__ == "__main__":
    main()
