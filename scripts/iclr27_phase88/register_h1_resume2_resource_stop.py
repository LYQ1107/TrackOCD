#!/usr/bin/env python3
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main() -> None:
    p = OUT / "checkpoints/h1_false_merge_reset_resume2_f3_step028000.pt"
    event = {
        "schema_version": "trackocd.phase88.resource_event.v1",
        "phase": 88,
        "route": "H1_FALSE_MERGE_RESET_RESUME2",
        "status": "PAUSED_RESOURCE_RESUME_REQUIRED",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": "available RAM fell to about 28 GiB, below mandatory 25 percent floor of about 31.25 GiB during f3 continuation",
        "task_owned_pids": [39627, 39629, 18522],
        "termination": "explicit SIGTERM; external processes untouched",
        "latest_valid_checkpoint": {"path": str(p.resolve()), "step": 28000, "sha256": sha(p)},
        "preserved_markers": ["outputs/iclr27_phase88/completion/h1_false_merge_reset_resume2_f3.launched"],
        "next_action": "resume f3 from step28000 with one worker after RAM preflight",
        "public_dev_q1_sealed_accessed": False,
    }
    out = OUT / "audit/h1_resume2_resource_stop.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(out)

if __name__ == "__main__":
    main()
