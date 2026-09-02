#!/usr/bin/env python3
"""Record the bounded exact-evaluator timeout and its minimal repair."""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "outputs/iclr27_phase76ar/audit/repair_events.json"


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def main() -> None:
    existing = json.loads(PATH.read_text()) if PATH.exists() else {"phase": "Phase76AR", "events": []}
    event = {
        "attempt": 1, "stage": "exact_train_disjoint_validation", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": "PYTHONPATH=. .../evaluate_exact.py", "exit_code": 143,
        "traceback": "bounded tool wait returned SIGTERM after 300s; no JSON/temporary artifact was produced",
        "root_cause": "single invocation evaluated all four folds and both streams, exceeding one bounded shell window",
        "repair": "add --fold/--stream partial atomic artifacts and --aggregate combiner; preserve all inference math",
        "protocol_changed": False, "artifact_changed": False, "resume_point": "fold-scoped exact replay from frozen best checkpoints",
        "external_process_terminated": False, "public_or_sealed_access": False,
    }
    if not any(x.get("command") == event["command"] and x.get("exit_code") == 143 for x in existing.get("events", [])):
        existing.setdefault("events", []).append(event)
    atomic(PATH, existing); print(json.dumps(existing, sort_keys=True))


if __name__ == "__main__": main()
