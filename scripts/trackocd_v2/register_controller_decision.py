#!/usr/bin/env python3
"""Register the single allowed v2 controller comparison before execution.

The stage is deliberately a protocol record, not a metric-based selection.
Nearest is the frozen incumbent; the only candidate opened downstream is the
conservative probability-hierarchy controller.  Val selection remains a
separate stage and Test is not read here.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


OUTPUT = OUTPUT_TARGET / "audit/controller_decision.json"
GT_BASELINE = OUTPUT_TARGET / "tables/gt_nearest.json"
SAFE_CONTROLLER_CODE = ROOT / "src/trackocd_v2/methods/safe_controller.py"


def main() -> int:
    out = ensure_output_layout()
    if not GT_BASELINE.is_file():
        raise FileNotFoundError(GT_BASELINE)
    payload = {
        "schema_version": "trackocd.v2.controller_decision.v1",
        "status": "CONTROLLER_CANDIDATES_REGISTERED",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "incumbent": {
            "name": "NearestPrototype",
            "role": "frozen vocabulary-free baseline",
            "source": str(GT_BASELINE.resolve()),
            "source_sha256": sha256_file(GT_BASELINE),
        },
        "candidate": {
            "name": "SafePersistentController",
            "role": "single conservative probability-hierarchy candidate",
            "source": str(SAFE_CONTROLLER_CODE.resolve()),
            "source_sha256": sha256_file(SAFE_CONTROLLER_CODE),
            "probability_factorization": "P(DEFER)=P(WAIT); P(KNOWN)=P(READY)*P(KNOWN-router)*P(KNOWN-k); P(EXISTING/NEW)=P(READY)*P(OPEN-router)*P(EXISTING/NEW|OPEN)",
        },
        "selection_is_not_done": True,
        "selection_deferred_to": "Val final selection after formal predicted-stream replay",
        "forbidden_inputs": ["category_text", "novel_category_label", "future_observation", "GT_at_inference", "physical_id_as_feature", "StateMemory_feedback"],
        "test_semantic_accessed": False,
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
