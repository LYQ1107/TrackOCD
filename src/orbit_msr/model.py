"""ORBIT-MSR model: shared adapter + memory-scale-robust factorized heads."""
from __future__ import annotations

from src.orbit_fc.model import ORBITFCModel


class ORBITMSRModel(ORBITFCModel):
    """Identical architecture to ORBIT-FC; training/evaluation differ."""

    pass
