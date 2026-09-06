#!/usr/bin/env python3
"""Final lightweight Phase87 artifact/process integrity check."""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase87"


def main() -> None:
    json_errors = []
    json_count = 0
    for path in OUT.rglob("*.json"):
        try:
            json.loads(path.read_text()); json_count += 1
        except Exception as exc:
            json_errors.append({"path": str(path), "error": repr(exc)})
    required = [
        OUT / "audit" / "phase87_c0_decision.json", OUT / "audit" / "phase87_repair1_decision.json",
        OUT / "audit" / "phase87_c1_decision.json", OUT / "audit" / "contract_checks.json",
        ROOT / "docs" / "iclr27_phase87" / "PHASE87_AUTONOMOUS_RESEARCH_REPORT.md",
    ]
    done_units = sorted(str(p.relative_to(OUT)) for p in (OUT / "completion").glob("*.done"))
    forbidden_files = [str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file() and any(tok in p.name.lower() for tok in ("public_new", "q1_label", "sealed_label", "devplus"))]
    processes = []
    for proc in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if "iclr27_phase87" in cmd and "integrity_check.py" not in cmd:
            processes.append({"pid": int(proc.name), "cmd": cmd.strip()})
    payload = {
        "phase": 87,
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "json_count": json_count,
        "json_errors": json_errors,
        "required_artifacts_exist": all(p.exists() for p in required),
        "missing_artifacts": [str(p) for p in required if not p.exists()],
        "done_units": done_units,
        "checkpoint_root_exists": pathlib.Path("/data2/usr_for_deadline/trackocd_phase87/project_outputs/checkpoints").exists(),
        "output_symlink_valid": OUT.is_symlink() and OUT.resolve().exists(),
        "forbidden_output_files": forbidden_files,
        "residual_phase87_processes": processes,
        "report_nonempty": required[-1].exists() and required[-1].stat().st_size > 0,
        "status": "PASS" if not json_errors and all(p.exists() for p in required) and not processes and not forbidden_files and OUT.resolve().exists() else "FAIL",
    }
    target = OUT / "audit" / "integrity_check.json"; target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); tmp.replace(target)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
