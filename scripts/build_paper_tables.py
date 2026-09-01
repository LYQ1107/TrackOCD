#!/usr/bin/env python3
"""Build ICLR27 paper tables from frozen CSVs."""
import sys
sys.path.insert(0, "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_closure.phase1 import (
    protocol_statistics, legacy_vs_corrected, gt_track_tables, claim_evidence,
)
protocol_statistics()
legacy_vs_corrected()
gt_track_tables()
claim_evidence()
print("paper tables built")
