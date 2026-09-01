"""Read-only Phase29 facade over the frozen Phase26 TRAIN feature protocol."""
from __future__ import annotations

from pathlib import Path

from src.iclr27_phase26.protocol import (  # noqa: F401
    CSV_PATH,
    FEAT_PATH,
    by_track,
    load_aligned_features,
    order_key,
)

ROOT = Path(__file__).resolve().parents[2]
FOLD_MANIFEST = ROOT / "outputs/iclr27_phase29/manifests/fold_manifest.json"
PHASE26_DECISION = ROOT / "outputs/iclr27_phase26/audit/phase26_decision.json"
PHASE28_DECISION = ROOT / "outputs/iclr27_phase28/audit/phase28_decision.json"
POSITIVE_EVENTS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
PREFIXES = (1, 2, 4, 8, 16)
POSITIVE_EVENT_DENOMINATOR = 76

__all__ = [
    "CSV_PATH", "FEAT_PATH", "FOLD_MANIFEST", "PHASE26_DECISION",
    "PHASE28_DECISION", "POSITIVE_EVENTS", "PREFIXES",
    "POSITIVE_EVENT_DENOMINATOR", "by_track", "load_aligned_features",
    "order_key",
]
