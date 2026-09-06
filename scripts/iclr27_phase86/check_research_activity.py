#!/usr/bin/env python3
"""One-shot Phase86 activity audit; never sleeps or finalizes."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.iclr27_phase86.execution_policy import parse_utc, remaining_seconds
OUT = ROOT / "outputs/iclr27_phase86"

def main() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    files = []
    for d in (OUT / "audit", OUT / "metrics", OUT / "completion"):
        if d.exists(): files.extend(p for p in d.rglob("*") if p.is_file())
    latest = max((p.stat().st_mtime for p in files), default=0.0)
    age = now.timestamp() - latest if latest else None
    unfinished = []
    for marker in OUT.rglob("*.launched") if OUT.exists() else []:
        if not marker.with_suffix(".done").exists() and not marker.with_suffix(".failed").exists(): unfinished.append(str(marker))
    rem = remaining_seconds()
    status = "RESEARCH_IDLE" if rem > 3600 and age is not None and age > 2700 else "ACTIVE"
    print(json.dumps({
        "schema_version": "trackocd.phase86.activity.v1",
        "now_utc": now.isoformat(), "remaining_seconds": rem,
        "status": status, "latest_artifact_age_seconds": age,
        "unfinished_markers": unfinished,
        "next_action": "continue highest-information unresolved Phase86 route" if status == "RESEARCH_IDLE" else "continue registered work or finalize only in unlocked interval",
        "finalization_allowed": rem <= 2700,
        "public_dev_q1_sealed_accessed": False,
    }, indent=2, sort_keys=True))

if __name__ == "__main__": main()
