#!/usr/bin/env python3
"""Record the bounded Phase76S resource and sealed-boundary preflight."""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76s/audit/resource_preflight.json"


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def main() -> None:
    payload = {
        "phase": "Phase76S",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "free_h": run(["free", "-h"]),
        "process_count": int(run(["bash", "-lc", "ps -e --no-headers | wc -l"]) or 0),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]),
        "disk_data1": shutil.disk_usage("/data1")._asdict(),
        "disk_data2": shutil.disk_usage("/data2")._asdict(),
        "gpu_mapping": {"fold0": 4, "fold1": 5, "fold2": 6, "fold3": 7},
        "estimated_worker_peak_rss_gib": 2.0,
        "ram_safety_floor": ">=25% available before/after launch",
        "external_gpus_left_untouched": [0, 1],
        "public_or_sealed_access": False,
    }
    atomic(OUT, payload); print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__": main()
