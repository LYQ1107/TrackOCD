#!/usr/bin/env python3
"""Register the Phase82R+ window and immutable input lineage."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase82r"
Q0 = ROOT / "outputs/iclr27_phase4t/train_stream/teta/tao_track.json"
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl")
START = "2026-09-04T02:17:53.672000+00:00"
DEADLINE = "2026-09-04T12:17:53.672000+00:00"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    registration = {
        "schema_version": "trackocd.phase82r.registration.v1",
        "phase": "Phase82R+",
        "start_time_utc": START,
        "deadline_utc": DEADLINE,
        "registered_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": branch,
        "q0_train_path": str(Q0),
        "q0_train_sha256": sha(Q0),
        "native_lineage_path": str(NATIVE),
        "native_lineage_sha256": sha(NATIVE),
        "phase82p_report": str(ROOT / "docs/iclr27_phase82p/PHASE82P_CAUSAL_PHYSICAL_ASSOCIATION_REPORT.md"),
        "phase82p_strict_o": "25/76 p16 both-reliable",
        "gpus": [4, 5, 6, 7],
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_as_model_input": False,
    }
    atomic(OUT / "audit/phase82r_registration.json", registration)
    atomic(OUT / "audit/status.json", registration)
    (OUT / "completion").mkdir(parents=True, exist_ok=True)
    (OUT / "completion/phase82r_registered.done").write_text("registered\n", encoding="utf-8")
    print(json.dumps(registration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
