#!/usr/bin/env python3
"""Record a Phase86 continuation without re-registering its original window."""
from __future__ import annotations
import datetime as dt
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase86"
REG = OUT / "audit/window_registration.json"
TARGET = OUT / "audit/resume_after_premature_finalization.json"

def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def parse(v: str) -> dt.datetime:
    return dt.datetime.fromisoformat(v.replace("Z", "+00:00"))

def main() -> None:
    registration = json.loads(REG.read_text(encoding="utf-8"))
    t = now()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    deadline = parse(registration["deadline_utc"])
    start = parse(registration["start_utc"])
    result = {
        "schema_version": "trackocd.phase86.resume.v1",
        "phase": 86,
        "resume_head": head,
        "resume_time_utc": t.isoformat().replace("+00:00", "Z"),
        "original_start_utc": registration["start_utc"],
        "original_deadline_utc": registration["deadline_utc"],
        "elapsed_seconds": (t - start).total_seconds(),
        "remaining_seconds": (deadline - t).total_seconds(),
        "premature_report_head": "e9801f9ca34fd135a02813b5709ae3c78d3ea8e7",
        "reason": "Phase86 was prematurely finalized after roughly 40 minutes although the registered ten-hour window remained open.",
        "do_not_reregister_window": True,
        "old_report_is_interim": True,
        "public_dev_q1_sealed_accessed": False,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    tmp = TARGET.with_name(TARGET.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(TARGET)
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
