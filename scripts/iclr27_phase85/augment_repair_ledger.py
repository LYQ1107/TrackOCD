#!/usr/bin/env python3
"""Append Phase85 execution/repair events to the immutable issue ledger."""
from __future__ import annotations
import datetime as dt
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "outputs/iclr27_phase85/audit/repair_events.json"


def atomic(path: Path, value: object) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def main() -> None:
    z = json.loads(PATH.read_text(encoding="utf-8"))
    events = list(z.get("events", []))
    events.extend([
        {"stage": "support_event_replay_selective_source_v1", "failure": "system Python lacked torch", "root_cause": "runtime mismatch, not model failure", "repair": "rerun with audited /home/lwr/anaconda3/envs/ovtr/bin/python", "protocol_changed": False, "artifact_changed": False, "task_pid": None},
        {"stage": "support_selection_audit", "failure": "direct invocation could not import project src", "root_cause": "missing project-root sys.path insertion", "repair": "minimal path insertion and py_compile, then same audit", "protocol_changed": False, "artifact_changed": False, "task_pid": None},
        {"stage": "selective_physical_replay_initial", "failure": "per-row Torch NumPy bridge replay was killed after profiling showed avoidable CPU/GPU overhead", "root_cause": "implementation performance bottleneck, no scientific artifact", "repair": "use exact NumPy MLP forward for frozen gate; smoke and targeted regression passed before formal replay", "protocol_changed": False, "artifact_changed": False, "task_pids": [32861, 32862], "termination": "explicit SIGTERM to task-owned PIDs only; no external process touched"},
        {"stage": "physical_gate_smoke_r1", "failure": "evaluation used truthiness of NumPy id array", "root_cause": "scalar/array API mismatch", "repair": "replace if not ids with len(ids)==0; fresh smoke_r2 and targeted_r1 passed", "protocol_changed": False, "artifact_changed": False, "failed_marker": "physical_gate_smoke_r1_f0.launched"},
        {"stage": "selective_source_cache", "failure": "none", "root_cause": "registered P1+S0 diagnostic", "repair": "causal single-root source cache with exact join and atomic NPZ", "protocol_changed": False, "artifact_changed": True, "result": "support replay negative; retained"},
    ])
    z.update({"schema_version": "trackocd.phase85.repair_events.v2", "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "events": events, "resource_events": list(z.get("resource_events", [])) + [{"event": "Phase85 selective replay performance repair", "oom": False, "external_process_terminated": False, "task_owned_pids": [32861, 32862], "note": "explicit SIGTERM after profiling; NumPy frozen-forward replay completed"}]})
    atomic(PATH, z)
    print(json.dumps({"status": "DONE", "events": len(events), "path": str(PATH.resolve())}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
