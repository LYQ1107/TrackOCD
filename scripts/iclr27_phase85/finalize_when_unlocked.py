#!/usr/bin/env python3
"""Finalize only inside the registered Phase85 finalization interval.

Unlike the Phase84 helper this process never sleeps.  Starting it early is an
explicit error so the research budget cannot be replaced by idle waiting.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "outputs/iclr27_phase85/audit"
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


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    reg = json.loads(REG.read_text(encoding="utf-8"))
    deadline = dt.datetime.fromisoformat(str(reg["deadline_utc"]).replace("Z", "+00:00"))
    target = deadline - dt.timedelta(minutes=45)
    remaining = (target - dt.datetime.now(dt.timezone.utc)).total_seconds()
    if remaining > 0:
        raise SystemExit("FINALIZATION_TOO_EARLY_RESEARCH_MUST_CONTINUE")
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    lock.update({"allowed": True, "reason": "registered deadline-minus-45-minute finalization interval reached", "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "deadline_utc": reg["deadline_utc"]})
    atomic_json(LOCK, lock)
    subprocess.run([PYTHON, "scripts/iclr27_phase85/generate_live_ledger.py"], cwd=ROOT, check=True)
    subprocess.run([PYTHON, "scripts/iclr27_phase85/generate_final_report.py"], cwd=ROOT, check=True)
    report = ROOT / "docs/iclr27_phase85/PHASE85_AUTONOMOUS_RESEARCH_REPORT.md"
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError("final report missing or empty")
    for p in (AUDIT / "phase85_decision.json", AUDIT / "report_provenance.json", AUDIT / "research_ledger.json", AUDIT / "repair_events.json"):
        json.loads(p.read_text(encoding="utf-8"))
    # The report and live note are small tracked artifacts; all large data
    # remains on the existing /data2 symlink target.
    subprocess.run(["git", "add", "-f", "docs/iclr27_phase85/PHASE85_AUTONOMOUS_RESEARCH_REPORT.md", "docs/iclr27_phase85/PHASE85_LIVE_LEDGER.md"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", "phase85 finalize autonomous research report"], cwd=ROOT, check=False)
    final_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
    # The report is generated before its finalization commit, so record the
    # exact repository head separately in the machine decision and lock.  This
    # keeps SCIENCE_HEAD, REPORT_GENERATION_HEAD and FINAL_REPOSITORY_HEAD
    # auditable without rewriting the report after the commit.
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    lock.update({"final_repository_head": final_head, "finalized_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    atomic_json(LOCK, lock)
    decision_path = AUDIT / "phase85_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision.update({"final_repository_head": final_head, "finalization_commit": final_head, "decision_sha256_before_head_update": sha(decision_path)})
    atomic_json(decision_path, decision)
    print(json.dumps({"status": "FINALIZED", "report": str(report.resolve()), "lock": str(LOCK.resolve()), "git_head": final_head}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
