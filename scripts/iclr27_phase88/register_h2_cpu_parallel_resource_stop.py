#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    names = {
        "0": "h2_known_suppression_cpu_formal_f0_step024000.pt",
        "1": "h2_known_suppression_cpu_formal_f1_step022000.pt",
        "2": "h2_known_suppression_cpu_formal_f2_step020000.pt",
        "3": "h2_known_suppression_cpu_formal_f3_step022000.pt",
    }
    checkpoints = {}
    for fold, name in names.items():
        path = OUT / "checkpoints" / name
        checkpoints[fold] = {
            "path": str(path.resolve()),
            "step": int(name.split("step")[-1].split(".")[0]),
            "sha256": sha256(path),
        }
    event = {
        "schema_version": "trackocd.phase88.resource_event.v1",
        "phase": 88,
        "route": "H2_KNOWN_SUPPRESSION_CPU_FORMAL",
        "status": "PAUSED_RESOURCE_PARALLEL_CPU_STOP",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": "four CPU workers drove MemFree to about 7.8 GiB, far below the mandatory 25 percent RAM floor, despite low per-process smoke RSS",
        "mem_total_kb": 131669916,
        "mem_free_before_stop_kb": 8185712,
        "mem_available_before_stop_kb": 87381740,
        "task_owned_pids": [20205, 20206, 20207, 20208, 20203, 18088],
        "termination": "explicit SIGTERM; no external process touched",
        "workers": 4,
        "latest_valid_checkpoints": checkpoints,
        "preserved_markers": [
            "outputs/iclr27_phase88/completion/h2_known_suppression_cpu_formal_f0.launched",
            "outputs/iclr27_phase88/completion/h2_known_suppression_cpu_formal_f1.launched",
            "outputs/iclr27_phase88/completion/h2_known_suppression_cpu_formal_f2.launched",
            "outputs/iclr27_phase88/completion/h2_known_suppression_cpu_formal_f3.launched",
        ],
        "next_action": "resume only with one CPU worker after MemFree recovers above the floor; do not relaunch four-way CPU parallelism",
        "public_dev_q1_sealed_accessed": False,
    }
    out = OUT / "audit/h2_cpu_parallel_resource_stop.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(out)


if __name__ == "__main__":
    main()
