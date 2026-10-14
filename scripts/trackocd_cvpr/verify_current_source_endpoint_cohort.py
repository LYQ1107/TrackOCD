#!/usr/bin/env python3
"""Run the registered public-only current-source endpoint cohort audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration
from src.trackocd_cvpr.current_source_endpoint_cohort import (
    AUDIT_OUTPUT, CONFIG, audit_cohort, configuration,
)

SCRIPT = 'scripts/trackocd_cvpr/verify_current_source_endpoint_cohort.py'
MODULE = 'src/trackocd_cvpr/current_source_endpoint_cohort.py'
SOURCES = (CONFIG, SCRIPT, MODULE,
           'configs/trackocd_cvpr/M_current_source_endpoint_refresh_transport_retry1_v1.json',
           'configs/trackocd_cvpr/M_current_source_endpoint_refresh_v1.json')


def source_registration(args):
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit,
                                args.git_proxy, allow_public_api=True)
    for path in SOURCES:
        if subprocess.check_output(
                ['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Audit source is not registered at preregistration commit: ' + path)
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    parser.add_argument('--git-proxy', default='')
    args = parser.parse_args()
    configuration(ROOT)
    proof = source_registration(args)
    result = audit_cohort(ROOT)
    result['live_remote_registration_proof'] = proof
    result['audit_preregistration_commit'] = args.preregistration_commit
    result['audit_source_sha256'] = {path: sha256_file(ROOT / path) for path in SOURCES}
    atomic_json(ROOT / AUDIT_OUTPUT, result)
    print(json.dumps({'status': result['status'], 'receipt': AUDIT_OUTPUT,
                      'case_count': result['case_count']}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
