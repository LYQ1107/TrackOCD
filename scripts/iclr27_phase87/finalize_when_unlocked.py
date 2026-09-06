#!/usr/bin/env python3
"""Allow finalization only after registered routes are complete or a hard blocker."""
from __future__ import annotations

import datetime as dt
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase87" / "audit" / "finalization_guard.json"


def main() -> None:
    c0 = OUT.parent / "phase87_c0_decision.json"
    r1 = OUT.parent / "phase87_repair1_decision.json"
    c1 = OUT.parent / "phase87_c1_decision.json"
    complete = c0.exists() and r1.exists() and c1.exists()
    payload = {"phase": 87, "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "registered_routes_complete": complete, "hard_blocker": "C1 formal gate failed; no registered downstream route is legal" if complete else None, "early_report_allowed": bool(complete), "sealed_or_public_run": False, "status": "UNLOCKED_HARD_BLOCKER" if complete else "LOCKED"}
    OUT.parent.mkdir(parents=True, exist_ok=True); tmp = OUT.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); tmp.replace(OUT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
