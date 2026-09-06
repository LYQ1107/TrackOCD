#!/usr/bin/env python3
"""Run Phase86 final report only after the registered deadline-minus-45m window."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.iclr27_phase86.execution_policy import assert_finalization_allowed

def main() -> None:
    assert_finalization_allowed()
    report = ROOT / "docs/iclr27_phase86/PHASE86_AUTONOMOUS_RESEARCH_REPORT.md"
    # The guarded report generator is invoked by the operator in the unlocked interval.
    import subprocess
    subprocess.run([sys.executable, "scripts/iclr27_phase86/generate_resume_report.py"], cwd=ROOT, check=True)
    if not report.is_file() or report.stat().st_size == 0: raise RuntimeError("missing final report")
    print(json.dumps({"status":"FINALIZATION_UNLOCKED","report":str(report.resolve())}, indent=2))

if __name__ == "__main__": main()
