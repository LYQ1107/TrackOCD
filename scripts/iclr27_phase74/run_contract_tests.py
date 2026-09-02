#!/usr/bin/env python3
"""Run Phase74 contract tests and the legacy evaluator once, recording exit codes."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase74"


def run(cmd: list[str]) -> dict:
    start = time.time()
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"command": " ".join(cmd), "cwd": str(ROOT), "start_epoch": start, "end_epoch": time.time(), "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "task_owned_pid": None}


def main() -> None:
    env = [sys.executable, "-m", "pytest", "-q", "tests/phase74"]
    old = [sys.executable, "tests/test_trackocd_evaluator.py"]
    recs = [run(env), run(old)]
    (OUT / "tests").mkdir(parents=True, exist_ok=True)
    (OUT / "tests/pytest_result.json").write_text(json.dumps(recs[0], indent=2) + "\n")
    (OUT / "tests/legacy_evaluator_direct.json").write_text(json.dumps(recs[1], indent=2) + "\n")
    log = OUT / "logs/commands.jsonl"; log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        for r in recs: f.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps({"pytest_exit": recs[0]["exit_code"], "legacy_exit": recs[1]["exit_code"]}))
    raise SystemExit(0 if all(r["exit_code"] == 0 for r in recs) else 1)


if __name__ == "__main__": main()
