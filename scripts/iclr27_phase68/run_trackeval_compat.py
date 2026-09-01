#!/usr/bin/env python3
"""Compatibility launcher for the vendored TrackEval TAO evaluator.

The pinned evaluator predates NumPy 1.24 and references ``np.int``.  Adding
the alias in this process keeps the third-party checkout read-only while
allowing the official evaluator to run unchanged.
"""
import runpy
import sys

import numpy as np

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

runpy.run_path("third_party/TrackEval/scripts/run_tao.py", run_name="__main__")
