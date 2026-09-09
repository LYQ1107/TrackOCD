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
        "route": "H2_KNOWN_SUPPRESSION_RESUME2",
        "status": "PAUSED_RESOURCE_NO_NEW_CHECKPOINT",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": "one-worker formal continuation reduced MemFree to about 25.5 GiB, below the mandatory 25 percent RAM floor, before the first 2000-step checkpoint",
        "mem_total_kb": 131669916,
        "mem_free_before_stop_kb": 26752036,
        "mem_available_before_stop_kb": 104054236,
        "task_owned_pids": [37167, 37165, 37161],
        "termination": "explicit SIGTERM; no external process touched",
        "workers": 1,
        "latest_valid_checkpoint": {
            "path": str((OUT / "checkpoints/h2_known_suppression_resume1_f0_step022000.pt").resolve()),
            "step": 22000,
        },
        "preserved_markers": [
            "outputs/iclr27_phase88/completion/h2_known_suppression_resume2_f0.launched",
        ],
        "smoke_and_targeted_passed": [
            "h2_known_suppression_resume2_smoke_f0",
            "h2_known_suppression_resume2_targeted_f0",
        ],
        "next_action": "do not relaunch formal H2 under the current host memory load; classify formal H2 as resource-blocked unless a new isolated RAM-safe window is authorized",
        "public_dev_q1_sealed_accessed": False,
    }
    out = OUT / "audit/h2_resume2_resource_stop.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(out)


if __name__ == "__main__":
    main()
