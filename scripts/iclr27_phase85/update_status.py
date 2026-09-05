#!/usr/bin/env python3
"""Write the current Phase85 stage status without changing scientific artifacts."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase85"


def sha(path: Path) -> str | None:
    if not path.is_file(): return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def main() -> None:
    status = {
        "schema_version": "trackocd.phase85.status.v2", "phase": "Phase85", "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stages": {"P0_issue_audit": "DONE", "P1_temporal_mean": "DONE", "P2_trackeval": "DONE", "P3_q0_parity": "PASS", "P4_single_anchor": "DONE", "P5_physical_to_R": "FAIL_DIAGNOSTIC", "B85S_reranker": "DONE_FAIL_SAFETY", "B85S_defer": "DONE_FAIL_SAFETY", "B85S_selective_source": "DONE_FAIL", "B85A_alignment": "NOT_AUTHORIZED", "C85_controller": "NOT_RUN", "sealed": "NOT_RUN"},
        "metrics": {"physical_temporal": str((OUT / "metrics/physical_r_temporal_comparison_v2.json").resolve()), "physical_selective": str((OUT / "metrics/physical_r_selective_comparison.json").resolve()), "support": str((OUT / "metrics/support_event_replay.json").resolve()), "support_selective_source": str((OUT / "metrics/support_event_replay_selective_source_v1.json").resolve()), "support_raw_defer": str((OUT / "metrics/support_event_replay_raw_defer_v1.json").resolve())},
        "artifact_hashes": {name: sha(OUT / rel) for name, rel in {"physical_temporal": "metrics/physical_r_temporal_comparison_v2.json", "physical_selective": "metrics/physical_r_selective_comparison.json", "support": "metrics/support_event_replay.json", "support_selective_source": "metrics/support_event_replay_selective_source_v1.json", "support_raw_defer": "metrics/support_event_replay_raw_defer_v1.json", "leakage_contract": "audit/leakage_contract.json"}.items()},
        "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "next_action": "finalize only inside registered deadline-minus-45-minute interval; no controller/alignment/backbone before gates"
    }
    atomic(OUT / "status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__": main()
