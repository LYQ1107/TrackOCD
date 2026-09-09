#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"


def main() -> None:
    event = {
        "schema_version": "trackocd.phase88.resource_event.v1",
        "phase": 88,
        "route": "H2_KNOWN_SUPPRESSION_CPU_SEQUENTIAL_RESUME1",
        "status": "PAUSED_RESOURCE_NO_NEW_CHECKPOINT",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": "one CPU worker continuation reduced MemFree to about 19.9 GiB before the first 2000-step resume checkpoint, below the mandatory 25 percent RAM floor",
        "mem_total_kb": 131669916,
        "mem_free_before_stop_kb": 19920612,
        "mem_available_before_stop_kb": 95427776,
        "task_owned_pids": [24805, 24803, 17486],
        "termination": "explicit SIGTERM; no external process touched",
        "workers": 1,
        "latest_valid_checkpoints": {
            "0": {
                "path": str((OUT / "checkpoints/h2_known_suppression_cpu_formal_f0_step024000.pt").resolve()),
                "step": 24000,
            },
            "1": {
                "path": str((OUT / "checkpoints/h2_known_suppression_cpu_formal_f1_step022000.pt").resolve()),
                "step": 22000,
            },
            "2": {
                "path": str((OUT / "checkpoints/h2_known_suppression_cpu_formal_f2_step020000.pt").resolve()),
                "step": 20000,
            },
            "3": {
                "path": str((OUT / "checkpoints/h2_known_suppression_cpu_formal_f3_step022000.pt").resolve()),
                "step": 22000,
            },
        },
        "preserved_markers": [
            "outputs/iclr27_phase88/completion/h2_known_suppression_cpu_resume1_f0.launched",
        ],
        "next_action": "wait for a memory-safe host/window; do not treat partial H2 as selected or formally validated",
        "public_dev_q1_sealed_accessed": False,
    }
    out = OUT / "audit/h2_cpu_sequential_resource_stop.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(out)


if __name__ == "__main__":
    main()
