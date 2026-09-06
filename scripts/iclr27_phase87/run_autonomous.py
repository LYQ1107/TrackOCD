#!/usr/bin/env python3
"""Machine-readable route ledger entry point (long runs use explicit supervisors)."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.iclr27_phase87.execution_policy import FOLD_GPU, ROUTES, FORMAL_GATE, WINDOW_DEADLINE_UTC


def main() -> None:
    out = ROOT / "outputs" / "iclr27_phase87" / "audit" / "autonomous_route_policy.json"
    payload = {"phase": 87, "routes": ROUTES, "fold_gpu": FOLD_GPU, "formal_gate": FORMAL_GATE, "deadline_utc": WINDOW_DEADLINE_UTC, "status": "POLICY_REGISTERED", "long_commands": "one bounded supervisor and one blocking wait", "sealed_inputs_forbidden": True}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); tmp.replace(out)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
