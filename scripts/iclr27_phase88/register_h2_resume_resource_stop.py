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
    path = OUT / "checkpoints/h2_known_suppression_resume1_f0_step022000.pt"
    event = {
        "schema_version": "trackocd.phase88.resource_event.v1",
        "phase": 88,
        "route": "H2_KNOWN_SUPPRESSION_RESUME1",
        "status": "PAUSED_RESOURCE_RESUME_REQUIRED",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": "MemFree fell to about 29.7 GiB, below the mandatory 25 percent RAM floor during one-worker continuation",
        "mem_total_kb": 131669916,
        "mem_free_before_stop_kb": 31129884,
        "mem_available_before_stop_kb": 107406128,
        "task_owned_pids": [22305, 22303, 22297],
        "termination": "explicit SIGTERM; no external process touched",
        "workers": 1,
        "latest_valid_checkpoints": {
            "0": {
                "path": str(path.resolve()),
                "step": 22000,
                "sha256": sha256(path),
            }
        },
        "preserved_markers": [
            "outputs/iclr27_phase88/completion/h2_known_suppression_resume1_f0.launched",
        ],
        "next_action": "do not select incomplete H2; resume only after a verified RAM-safe window, otherwise classify as resource-blocked",
        "public_dev_q1_sealed_accessed": False,
    }
    out = OUT / "audit/h2_resume1_resource_stop.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(out)


if __name__ == "__main__":
    main()
