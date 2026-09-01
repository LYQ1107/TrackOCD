#!/usr/bin/env python3
"""Capture the required Phase24 resource/sealing preflight."""
from __future__ import annotations
import json, os, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase24/audit/preflight.json"

def run(cmd: str) -> str:
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return (p.stdout + p.stderr).strip()

def main() -> None:
    value = {
        "command": "free -h; ps process summary; nvidia-smi; df -h /data1",
        "free_h": run("free -h"),
        "process_count": len(run("ps -e --no-headers").splitlines()),
        "phase24_processes_before": [x for x in run("ps -eo pid,ppid,etime,cmd").splitlines() if "iclr27_phase24" in x],
        "nvidia_smi": run("nvidia-smi"),
        "nvidia_query": run("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader"),
        "df_data1": run("df -h /data1"),
        "gpu_policy": "Phase24 uses GPUs 4,5,6,7 (0-3 occupied by unrelated InterMOT workers; GPU9 viewer untouched)",
        "ram_floor": "retain at least 25% system RAM",
        "public_q1_sealed": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{OUT.name}.", dir=str(OUT.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, OUT)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print(json.dumps({"process_count": value["process_count"], "nvidia_query": value["nvidia_query"], "df_data1": value["df_data1"]}, indent=2))

if __name__ == "__main__": main()
