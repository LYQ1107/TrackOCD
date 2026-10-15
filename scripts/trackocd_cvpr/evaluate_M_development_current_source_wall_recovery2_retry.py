#!/usr/bin/env python3
"""Fresh namespace retry after a pre-private-input remote transport failure."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import evaluate_M_development_current_source as current
from scripts.trackocd_cvpr import evaluate_M_development_current_source_wall_recovery2 as recovery2

SCRIPT = 'scripts/trackocd_cvpr/evaluate_M_development_current_source_wall_recovery2_retry.py'
RETRY_SUFFIX = '_retry1'


def configuration(root, args):
    cfg = recovery2.configuration(root, args)
    cfg['output_directory'] += RETRY_SUFFIX
    return cfg


def _apply_patch():
    recovery2._apply_patch()
    current.SCRIPT = SCRIPT
    current.SOURCES = tuple(dict.fromkeys((*current.SOURCES, SCRIPT)))
    current.configuration = configuration


def main():
    _apply_patch()
    return current.main()


if __name__ == '__main__':
    sys.exit(main())
