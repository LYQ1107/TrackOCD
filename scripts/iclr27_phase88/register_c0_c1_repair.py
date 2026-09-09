#!/usr/bin/env python3
"""Record the minimum continuation-contract repair without changing the route."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    prereg = OUT / "audit/c0_continue_vs_c1_support_preregistration.json"
    failed = [OUT / "completion" / f"c0_continue_f{i}.launched" for i in range(4)]
    payload = {
        "schema_version": "trackocd.phase88.c0_c1_repair.v1",
        "phase": 88,
        "repair_cycle": 1,
        "status": "MINIMAL_REPAIR_ACCEPTED",
        "first_actionable_root_cause": "resume checkpoint loaded with map_location=cuda; CPU torch RNG ByteTensor was mapped to CUDA and torch.set_rng_state rejected it",
        "fix": "load resume payload on CPU, then restore CPU RNG; model/optimizer state_dict loading remains unchanged",
        "failed_attempts_preserved": [str(x.resolve()) for x in failed],
        "failed_attempt_log_hashes": [sha(OUT / "logs" / f"c0_continue_f{i}.log") for i in range(4)],
        "smoke": {
            "c0_tag": "c0_continue_fix1_smoke_f0",
            "c0_metrics": str((OUT / "metrics/c0_continue_fix1_smoke_f0.json").resolve()),
            "c0_checkpoint_sha256": "f40e22ddb35a0234b18145a54af458fa578475c1c1c76f811a07b40003ec50a7",
            "c1_tag": "c1_support_fix1_smoke_f0",
            "c1_metrics": str((OUT / "metrics/c1_support_fix1_smoke_f0.json").resolve()),
            "c1_checkpoint_sha256": "c97e254fbeca1d066ab623db70e2a6db56651f78f3b281ce95e582a7e0101cc9",
            "status": "PASS",
        },
        "targeted": {
            "c0_tag": "c0_continue_fix1_targeted_f0",
            "metrics": str((OUT / "metrics/c0_continue_fix1_targeted_f0.json").resolve()),
            "checkpoint_sha256": "9c326f06fc14319ca2eb40084c2d010929724a871b6957daa147c35a5e0bdc16",
            "updates": 500,
            "status": "PASS",
        },
        "route_contract_unchanged": True,
        "selection_from_held": False,
        "public_dev_q1_sealed_accessed": False,
        "preregistration_sha256": sha(prereg),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    out = OUT / "audit/c0_c1_control_repair1.json"
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
