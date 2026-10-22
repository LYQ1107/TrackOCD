#!/usr/bin/env python3
"""Wait for both Val predicted baselines, then build their Val-only report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_v2_runtime"
DEFAULT_LOG = ROOT / "outputs/trackocd_cvpr/cvpr2/val_baseline_report_gate_20261022.log"
DEFAULT_REPORT = ROOT / "docs/trackocd_cvpr/VAL_PRED_BASELINE_REPORT_20261022.md"
METHODS = ("nearest", "dpmeans")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "COMPLETE"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _event(handle, event: str, **fields) -> None:
    payload = {"event": event, "time": _now(), **fields}
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()
    print(json.dumps(payload, sort_keys=True), flush=True)


def run(output_root: Path, report_path: Path, log_path: Path, sleep_seconds: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary = output_root / "tables/val_pred_baseline_summary.json"
    with log_path.open("a", encoding="utf-8") as handle:
        _event(handle, "REPORT_GATE_START", report=str(report_path.resolve()), test_semantic_accessed=False)
        last_state = None
        while True:
            state = {method: _complete(output_root / "tables" / f"pred_{method}.json") for method in METHODS}
            state_key = tuple(state[method] for method in METHODS)
            if state_key != last_state:
                _event(handle, "BASELINE_STATE", **state)
                last_state = state_key
            if all(state.values()):
                if _complete(summary):
                    _event(handle, "REPORT_ALREADY_COMPLETE", summary=str(summary.resolve()))
                    return 0
                command = [
                    sys.executable,
                    str((ROOT / "scripts/trackocd_v2/build_val_pred_baseline_report.py").resolve()),
                    "--output-root", str(output_root.resolve()),
                    "--report", str(report_path.resolve()),
                ]
                _event(handle, "REPORT_START", command=command)
                result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
                _event(handle, "REPORT_EXIT", returncode=result.returncode)
                return result.returncode
            time.sleep(sleep_seconds)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--sleep-seconds", type=int, default=60)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.sleep_seconds < 1:
        raise ValueError("sleep-seconds must be positive")
    return run(args.output_root.resolve(), args.report.resolve(), args.log.resolve(), args.sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
