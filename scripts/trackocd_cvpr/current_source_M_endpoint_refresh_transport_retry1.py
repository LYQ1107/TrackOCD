#!/usr/bin/env python3
"""Fresh current-source endpoint retry after a pre-input transport failure."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import current_source_M_endpoint_refresh_terminal_recovery1 as recovery

SCRIPT = 'scripts/trackocd_cvpr/current_source_M_endpoint_refresh_transport_retry1.py'
CONFIG = 'configs/trackocd_cvpr/M_current_source_endpoint_refresh_transport_retry1_v1.json'
FAILED = 'outputs/trackocd_cvpr/cvpr4/M_GT_window_current_source_v1_relative_quarantine_s1027'
RECOVERY_PREFIX = 'outputs/trackocd_cvpr/cvpr4/M_GT_window_current_source_v1_transport_retry1'
RECOVERY_TERMINAL = 'outputs/trackocd_cvpr/cvpr4/M_GT_window_current_source_v1_terminal_recovery1_relative_quarantine_s1027_supervisor_attempt_1.json'

def _apply_patch():
    state = (recovery.SCRIPT, recovery.CONFIG, recovery.FAILED,
             recovery.RECOVERY_PREFIX, recovery.base.SOURCES,
             recovery.base.original.SOURCES)
    recovery.SCRIPT = SCRIPT
    recovery.CONFIG = CONFIG
    recovery.FAILED = FAILED
    recovery.RECOVERY_PREFIX = RECOVERY_PREFIX
    recovery.base.SOURCES = tuple(dict.fromkeys((*recovery.base.SOURCES, SCRIPT, CONFIG,
        'scripts/trackocd_cvpr/current_source_M_endpoint_refresh_terminal_recovery1.py',
        'configs/trackocd_cvpr/M_current_source_endpoint_refresh_terminal_recovery_v1.json',
        RECOVERY_TERMINAL)))
    recovery.base.original.SOURCES = recovery.base.SOURCES
    return state


def _restore(state):
    (recovery.SCRIPT, recovery.CONFIG, recovery.FAILED, recovery.RECOVERY_PREFIX,
     recovery.base.SOURCES, recovery.base.original.SOURCES) = state


def configuration(args):
    state = _apply_patch()
    try:
        return recovery.configuration(args)
    finally:
        _restore(state)


def main():
    _apply_patch()
    return recovery.main()


if __name__ == '__main__':
    sys.exit(main())
