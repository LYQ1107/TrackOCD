#!/usr/bin/env python3
"""Write the CPU-only route manifest for the physical frontend bake-off.

This command records the route and evaluator contract.  It deliberately does
not import a detector framework, inspect GPU state, open images, or launch a
frontend.  Native inference can consume this manifest once resources pass the
separate preflight.
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

from src.trackocd_v2.frontend_contract import route_manifest  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = ensure_output_layout()
    destination = args.out or (out / "audit/frontend_bakeoff_route.json")
    payload: dict[str, Any] = route_manifest(ROOT)
    payload.update({
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository_root": str(ROOT.resolve()),
        "route_manifest_is_execution_plan_only": True,
        "canonical_val_annotation_sha256": sha256_file(ROOT / "data/raw/tao/annotations/validation.json"),
    })
    atomic_json(destination, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
