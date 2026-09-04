#!/usr/bin/env python3
"""Run the frozen Phase75B strict-O evaluator into Phase82R only."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "scripts/iclr27_phase82p/evaluate_strict_o_residual.py"
spec = importlib.util.spec_from_file_location("phase82r_strict_o_readonly", OLD)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {OLD}")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
_atomic = module.atomic_json
def _phase82r_atomic(path, value):
    path = Path(path)
    if "outputs/iclr27_phase82p" in str(path):
        path = Path(str(path).replace("outputs/iclr27_phase82p", "outputs/iclr27_phase82r"))
    return _atomic(path, value)
module.atomic_json = _phase82r_atomic
if __name__ == "__main__":
    module.main()
