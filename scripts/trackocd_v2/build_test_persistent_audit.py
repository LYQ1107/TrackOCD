#!/usr/bin/env python3
"""Register the frozen TAO Test persistent-OCD result.

Persistent and Standard OCD are emitted by the same evaluator run, but the
stage boundary is kept explicit so the supervisor cannot report a final
persistent result without first sealing the Test causal decisions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.protocol import assert_test_semantic_access_allowed  # noqa: E402


FINAL_FREEZE = OUTPUT_TARGET / "audit/FINAL_FREEZE.json"
TEST_OCD = OUTPUT_TARGET / "tables/test_ocd.json"
OUTPUT = OUTPUT_TARGET / "audit/test_persistent.json"


def build() -> dict[str, Any]:
    assert_test_semantic_access_allowed(FINAL_FREEZE, "register frozen TAO Test persistent OCD")
    if not TEST_OCD.is_file():
        raise FileNotFoundError(TEST_OCD)
    result = json.loads(TEST_OCD.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETE":
        raise RuntimeError(f"Test OCD result is not complete: {result.get('status')}")
    if result.get("test_selection_or_tuning") is not False:
        raise RuntimeError("Test OCD result does not prove selection/tuning was disabled")
    if result.get("causal_decisions_sealed_before_test_gt_join") is not True:
        raise RuntimeError("Test OCD result does not prove causal decisions preceded the GT join")
    payload = {
        "schema_version": "trackocd.v2.test_persistent_audit.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_test_ocd": str(TEST_OCD.resolve()),
        "source_test_ocd_sha256": sha256_file(TEST_OCD),
        "frontend": result.get("frontend"),
        "frontend_slug": result.get("frontend_slug"),
        "methods": result.get("methods"),
        "test_selection_or_tuning": False,
        "test_semantic_accessed": True,
        "final_freeze": str(FINAL_FREEZE.resolve()),
        "final_freeze_sha256": sha256_file(FINAL_FREEZE),
    }
    atomic_json(OUTPUT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ensure_output_layout()
    result = build()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
