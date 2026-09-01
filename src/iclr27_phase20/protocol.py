"""Small, auditable Phase20 protocol constants."""
from __future__ import annotations

from dataclasses import dataclass

from . import PREFIXES, RELIABLE_IOU


@dataclass(frozen=True)
class Phase20Protocol:
    prefixes: tuple[int, ...] = PREFIXES
    reliable_iou: float = RELIABLE_IOU
    positive_denominator: int = 76
    gate_o_majority: float = 0.50
    sealed_inputs: tuple[str, ...] = ("DEV+", "Q1", "public new-model labels")


DEFAULT = Phase20Protocol()

