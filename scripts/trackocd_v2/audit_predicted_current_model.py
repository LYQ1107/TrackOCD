#!/usr/bin/env python3
"""Record the current-model decision for the predicted-stream contract.

The frozen Phase90/H3 artifact is deliberately not replayed on predicted
tracks: its legacy raw and geometry inputs do not match the v2 feature
contract, and no legal physical-track adapter is available.  This audit is a
negative comparability record, not a model metric.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


GT_CONTRACT = OUTPUT_TARGET / "audit/current_model_contract.json"
OUTPUT = OUTPUT_TARGET / "audit/predicted_current_model_contract.json"


def main() -> int:
    out = ensure_output_layout()
    if not GT_CONTRACT.exists():
        raise FileNotFoundError(GT_CONTRACT)
    contract = json.loads(GT_CONTRACT.read_text(encoding="utf-8"))
    result = {
        "schema_version": "trackocd.v2.predicted_current_model_contract.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": "EXCLUDED_NOT_COMPARABLE",
        "metric_produced": False,
        "gt_contract_source": str(GT_CONTRACT.resolve()),
        "gt_contract_source_sha256": sha256_file(GT_CONTRACT),
        "gt_contract_decision": contract.get("decision"),
        "reason": "The frozen H3 checkpoint uses the legacy 0.8*CLS+0.2*ROI and Phase19R-normalized geometry contract; v2 predicted features are pure CLS plus separate v2 geometry, and no legal physical-track adapter exists.",
        "predicted_stream_model_input": False,
        "test_semantic_accessed": False,
    }
    atomic_json(out / OUTPUT.relative_to(OUTPUT_TARGET), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
