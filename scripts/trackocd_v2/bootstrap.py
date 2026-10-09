#!/usr/bin/env python3
"""Register the TrackOCD v2 namespace and output contract."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout  # noqa: E402
from src.trackocd_v2.protocol import STAGES  # noqa: E402


def main() -> None:
    out = ensure_output_layout()
    (out / "audit").mkdir(parents=True, exist_ok=True)
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "schema_version": "trackocd.v2.bootstrap.v1",
        "project": "TrackOCD-v2",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head": head,
        "git_branch": branch,
        "historical_namespace_read_only": ["Phase19R", "Phase88", "Phase89", "Phase90"],
        "output_target": str(OUTPUT_TARGET),
        "output_link": str(ROOT / "outputs/trackocd_v2"),
        "stages": list(STAGES),
        "test_semantic_policy": "structural audit before FINAL_FREEZE; semantic evaluation only after FINAL_FREEZE",
        "public_model_inputs_forbidden": [
            "novel category name", "novel text prompt", "novel text embedding",
            "novel category logits", "GT category id", "future rows", "semantic labels",
        ],
        "status": "REGISTERED",
    }
    atomic_json(out / "audit/bootstrap.json", payload)
    state = {
        "schema_version": "trackocd.v2.autonomous_state.v1",
        "state": "BOOTSTRAP",
        "status": "REGISTERED",
        "updated_utc": payload["created_utc"],
        "completed_stages": [],
        "pending_stages": list(STAGES[1:]),
        "artifacts": {"BOOTSTRAP": str((out / "audit/bootstrap.json").resolve())},
        "failure_history": [],
        "resource_events": [],
        "test_semantic_access": {"accessed": False, "reason": "FINAL_FREEZE absent"},
    }
    atomic_json(out / "audit/autonomous_state.json", state)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
