"""Read-only Phase27 correspondence protocol facade.

The only large inputs are the frozen Phase19R/Phase26 CSV and feature NPZ.
The fold manifest is exposed through a Phase27-local symlink so that all
outputs remain in the Phase27 namespace while the TRAIN-only split remains
byte-identical to the registered Phase22 split.
"""
from __future__ import annotations

from src.iclr27_phase26.protocol import (  # noqa: F401
    CSV_PATH,
    FEAT_PATH,
    by_track,
    load_aligned_features,
    order_key,
)

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLD_MANIFEST = ROOT / "outputs/iclr27_phase27/manifests/fold_manifest.json"
PHASE26_DECISION = ROOT / "outputs/iclr27_phase26/audit/phase26_decision.json"
PHASE26_FULL_EVENTS = ROOT / "outputs/iclr27_phase26/audit/full_76_event_summary.csv"
PREFIXES = (1, 2, 4, 8, 16)
POSITIVE_EVENT_DENOMINATOR = 76

__all__ = [
    "CSV_PATH", "FEAT_PATH", "FOLD_MANIFEST", "PHASE26_DECISION",
    "PHASE26_FULL_EVENTS", "PREFIXES", "POSITIVE_EVENT_DENOMINATOR",
    "by_track", "load_aligned_features", "order_key",
]
