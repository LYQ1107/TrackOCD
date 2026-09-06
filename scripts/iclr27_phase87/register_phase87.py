#!/usr/bin/env python3
"""Register the Phase87 execution window and immutable resource snapshot."""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase87" / "audit" / "window_registration.json"
WINDOW_HOURS = 10


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # resource snapshots should remain auditable
        return f"ERROR: {exc}"


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    deadline = now + dt.timedelta(hours=WINDOW_HOURS)
    head = run(["git", "rev-parse", "HEAD"])
    origin = run(["git", "ls-remote", "origin", "refs/heads/main"])
    gpu = run(["nvidia-smi"])
    ram = run(["free", "-h"])
    d1 = run(["df", "-h", "/data1"])
    d2 = run(["df", "-h", "/data2"])
    payload = {
        "schema_version": "trackocd.phase87.window.v1",
        "phase": 87,
        "start_time_utc": now.isoformat(),
        "deadline_utc": deadline.isoformat(),
        "window_hours": WINDOW_HOURS,
        "start_head": head,
        "origin_head": origin,
        "gpu_snapshot": gpu,
        "ram_snapshot": ram,
        "data1_snapshot": d1,
        "data2_snapshot": d2,
        "max_gpus": 4,
        "planned_gpu_ids": [5, 6, 7, 8],
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_text_or_category_as_model_input": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(OUT)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
