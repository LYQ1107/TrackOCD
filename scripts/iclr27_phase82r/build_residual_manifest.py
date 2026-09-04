#!/usr/bin/env python3
"""Phase82R namespace wrapper for the audited causal TRAIN manifest builder.

The implementation is imported read-only from Phase82P, while all manifests
and arrays are redirected to Phase82R and the corrected appearance cache.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
base = importlib.import_module("scripts.iclr27_phase82p.build_residual_manifest")
base.OUT_ROOT = ROOT / "outputs/iclr27_phase82r/manifests"
base.DATA_ROOT = Path("/data2/usr_for_deadline/trackocd_phase82r/data")
base.APPEARANCE = ROOT / "outputs/iclr27_phase82r/features/q0_dinov2_corrected_r1.npz"


if __name__ == "__main__":
    base.main()
