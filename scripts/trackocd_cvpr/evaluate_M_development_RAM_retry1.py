#!/usr/bin/env python3
"""Same qualified full frozen M dev, bounded namespace-only RAM recovery."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import evaluate_M_development_all_model_fingerprints as base
from src.trackocd_cvpr.M_development_RAM_recovery import recovery_configuration, SOURCES


def main():
    original_configuration = base.all_model_configuration
    base.SCRIPT = 'scripts/trackocd_cvpr/evaluate_M_development_RAM_retry1.py'
    base.SOURCES = tuple(dict.fromkeys((*base.SOURCES,*SOURCES)))
    def configuration(root, original, args):
        return recovery_configuration(root, original_configuration(root, original, args), args)
    base.all_model_configuration = configuration
    return base.main()


if __name__ == '__main__': sys.exit(main())
