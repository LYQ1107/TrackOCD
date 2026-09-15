"""Protocol constants and leakage guards for TrackOCD v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


STAGES = (
    "BOOTSTRAP",
    "DATA_AUDIT",
    "PROTOCOL_BUILD",
    "GT_FEATURE_BUILD",
    "GT_GEOMETRY_AUDIT",
    "GT_NEAREST",
    "GT_DPMEANS",
    "GT_PHE",
    "GT_CURRENT_MODEL",
    "GT_BENCHMARK_TABLE",
    "PREDICTED_STREAM_BUILD",
    "PRED_NEAREST",
    "PRED_DPMEANS",
    "PRED_PHE",
    "PRED_CURRENT_MODEL",
    "OCD_BASELINE_EXTENSIONS",
    "FRONTEND_AUDIT",
    "FRONTEND_SIMOWT",
    "FRONTEND_OVTR",
    "FRONTEND_COVTRACK_NATIVE",
    "FRONTEND_COVTRACK_NOSEM",
    "FRONTEND_SELECTION",
    "REPRESENTATION_DECISION",
    "SEMANTIC_ADAPTER",
    "CONTROLLER_DECISION",
    "SAFE_CONTROLLER",
    "VAL_FINAL_SELECTION",
    "FINAL_FREEZE",
    "TAO_TEST_STANDARD_OCD",
    "TAO_TEST_PERSISTENT",
    "TAO_TEST_TRACKING",
    "FINAL_TABLES",
    "FINAL_REPORT",
    "COMPLETE",
)


def load_ids(path: Path) -> set[int]:
    values = json.loads(path.read_text())
    return {int(value) for value in values}


def validate_roles(known: Iterable[int], novel: Iterable[int], distractor: Iterable[int]) -> None:
    known_set, novel_set, distractor_set = map(set, (known, novel, distractor))
    if known_set & novel_set:
        raise ValueError("known and novel category sets overlap")
    if known_set & distractor_set:
        raise ValueError("known and distractor category sets overlap")
    if novel_set & distractor_set:
        raise ValueError("novel and distractor category sets overlap")


def assert_test_semantic_access_allowed(final_freeze: Path | None, purpose: str) -> None:
    """Reject test semantic access before the explicit final freeze.

    Structural test audits use their own path and do not call this function.
    Any future semantic evaluator, selector, or threshold script must call it
    before loading Test labels.
    """

    if final_freeze is None or not final_freeze.exists():
        raise RuntimeError(f"TEST_SEMANTIC_LEAKAGE_FORBIDDEN: {purpose}")
