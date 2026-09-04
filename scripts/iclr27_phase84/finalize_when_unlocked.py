#!/usr/bin/env python3
"""Wait once for the registered Phase84 finalization interval, then finalize.

The waiting process is the single bounded supervisor for finalization.  It does
not start scientific work and never touches external jobs.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "outputs/iclr27_phase84/audit"
LOCK = AUDIT / "finalization_lock.json"
REG = AUDIT / "window_registration.json"
PYTHON = "/home/lwr/anaconda3/envs/ovtr/bin/python"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def main() -> None:
    reg = json.loads(REG.read_text(encoding="utf-8"))
    deadline = dt.datetime.fromisoformat(str(reg["deadline_utc"]).replace("Z", "+00:00"))
    target = deadline - dt.timedelta(minutes=45)
    while True:
        remaining = (target - dt.datetime.now(dt.timezone.utc)).total_seconds()
        if remaining <= 0:
            break
        time.sleep(min(remaining, 3600.0))
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    lock.update({"allowed": True, "reason": "registered deadline-minus-45-minute finalization interval reached", "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "deadline_utc": reg["deadline_utc"]})
    atomic_json(LOCK, lock)
    subprocess.run([PYTHON, "scripts/iclr27_phase84/generate_live_ledger.py"], cwd=ROOT, check=True)
    subprocess.run([PYTHON, "scripts/iclr27_phase84/generate_final_report.py"], cwd=ROOT, check=True)
    report = ROOT / "docs/iclr27_phase84/PHASE84_AUTONOMOUS_RESEARCH_REPORT.md"
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError("final report missing or empty")
    for p in (AUDIT / "phase84_decision.json", AUDIT / "report_provenance.json", AUDIT / "research_ledger.json", AUDIT / "repair_events.json", AUDIT / "validation_evidence_ledger.json"):
        json.loads(p.read_text(encoding="utf-8"))
    # The report and live note are small tracked artifacts; all large data
    # remains on the existing /data2 symlink target.
    subprocess.run(["git", "add", "-f", "docs/iclr27_phase84/PHASE84_AUTONOMOUS_RESEARCH_REPORT.md", "docs/iclr27_phase84/PHASE84_LIVE_LEDGER.md"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", "phase84 finalize autonomous research report"], cwd=ROOT, check=False)
    subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
    print(json.dumps({"status": "FINALIZED", "report": str(report.resolve()), "lock": str(LOCK.resolve()), "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
