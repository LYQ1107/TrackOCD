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
    checkpoints = {}
    for fold, step in ((0, 18000), (1, 22000), (2, 20000)):
        p = OUT / "checkpoints" / f"h1_false_merge_reset_f{fold}_step{step:06d}.pt"
        checkpoints[str(fold)] = {"path": str(p.resolve()), "step": step, "sha256": sha(p)}
    event = {
        "schema_version": "trackocd.phase88.resource_event.v1",
        "phase": 88,
        "route": "H1_FALSE_MERGE_RESET",
        "status": "PAUSED_RESOURCE_RESUME_REQUIRED",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": "available RAM fell to about 28 GiB, below mandatory 25 percent floor of about 31.25 GiB while three H1 workers were active",
        "task_owned_pids": [35717, 35718, 35721, 35722, 35723],
        "termination": "explicit SIGTERM; no external process touched",
        "workers": 3,
        "latest_valid_checkpoints": checkpoints,
        "preserved_markers": [
            "outputs/iclr27_phase88/completion/h1_false_merge_reset_f0.launched",
            "outputs/iclr27_phase88/completion/h1_false_merge_reset_f1.launched",
            "outputs/iclr27_phase88/completion/h1_false_merge_reset_f2.launched",
        ],
        "next_action": "resume each unfinished fold sequentially with new route tags and one bounded worker after smoke/targeted continuation",
        "public_dev_q1_sealed_accessed": False,
    }
    out = OUT / "audit/h1_resource_stop.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(out)

if __name__ == "__main__":
    main()
