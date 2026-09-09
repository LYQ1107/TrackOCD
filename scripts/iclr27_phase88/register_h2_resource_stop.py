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
    latest = {
        "0": "h2_known_suppression_f0_step018000.pt",
        "1": "h2_known_suppression_f1_step020000.pt",
        "2": "h2_known_suppression_f2_step018000.pt",
    }
    checkpoints = {}
    for fold, name in latest.items():
        path = OUT / "checkpoints" / name
        checkpoints[fold] = {
            "path": str(path.resolve()),
            "step": int(name.split("step")[-1].split(".")[0]),
            "sha256": sha256(path),
        }
    event = {
        "schema_version": "trackocd.phase88.resource_event.v1",
        "phase": 88,
        "route": "H2_KNOWN_SUPPRESSION",
        "status": "PAUSED_RESOURCE_RESUME_REQUIRED",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": "MemFree fell just below the mandatory 25 percent RAM floor while three H2 workers were active",
        "mem_total_kb": 131669916,
        "mem_free_before_stop_kb": 32642232,
        "mem_available_before_stop_kb": 104322720,
        "task_owned_pids": [30846, 30847, 30848, 30843, 30827],
        "termination": "explicit SIGTERM; no external process touched",
        "workers": 3,
        "latest_valid_checkpoints": checkpoints,
        "preserved_markers": [
            "outputs/iclr27_phase88/completion/h2_known_suppression_f0.launched",
            "outputs/iclr27_phase88/completion/h2_known_suppression_f1.launched",
            "outputs/iclr27_phase88/completion/h2_known_suppression_f2.launched",
        ],
        "next_action": "resume unfinished folds sequentially with new tags and one bounded worker after smoke/targeted continuation",
        "public_dev_q1_sealed_accessed": False,
    }
    out = OUT / "audit/h2_resource_stop.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(out)


if __name__ == "__main__":
    main()
