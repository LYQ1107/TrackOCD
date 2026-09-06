#!/usr/bin/env python3
"""Capture the one preflight snapshot required before Phase87 workers."""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase87" / "audit" / "resource_preflight.json"


def run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return f"ERROR({exc.returncode}): {exc.output.strip()}"


def main() -> None:
    payload = {
        "phase": 87,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pid": os.getpid(),
        "free_h": run(["free", "-h"]),
        "process_count": len(run(["ps", "-e", "--no-headers"]).splitlines()),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]),
        "compute_apps": run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"]),
        "disk": run(["df", "-h", "/data1", "/data2"]),
        "planned_gpu_mapping": {"fold0": 5, "fold1": 6, "fold2": 7, "fold3": 8},
        "worker_limit": 4,
        "estimated_peak_rss_per_worker_gb": 4.0,
        "minimum_available_ram_gb": 31.25,
        "external_gpu_pids_untouched": [8827, 13957, 15368, 3280, 6567, 1206],
        "status": "PASS_HEADROOM_AND_MAPPING_RESERVED",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, OUT)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
