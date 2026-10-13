#!/usr/bin/env python3
"""Registered public-only completion observation audit, no worker restart."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration
from src.trackocd_cvpr.C_terminal_counter_qualification import qualified_completed_C

CONFIG = 'configs/trackocd_cvpr/C_terminal_counter_readback_v1.json'
DIRECTORY = 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Top1_conditioned_s1027'
SOURCES = (CONFIG, 'scripts/trackocd_cvpr/qualify_C_terminal_readback.py',
    'src/trackocd_cvpr/C_terminal_counter_qualification.py', 'src/trackocd_cvpr/remote_registration.py',
    'src/trackocd_v2/io.py', 'configs/trackocd_cvpr/C_GT_causal_development_v1.json',
    DIRECTORY + '/launch.json', DIRECTORY + '/result.json', DIRECTORY + '_supervisor_attempt_1.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True); parser.add_argument('--git-proxy', default='')
    args = parser.parse_args()
    proof = verify_registration(ROOT, 'research/cvpr-trackocd-autonomous-v2', args.preregistration_commit,
                                args.git_proxy, allow_public_api=True)
    for path in SOURCES:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Actual registered readback source and immutable public evidence')
    cfg = json.loads((ROOT / CONFIG).read_text()); directory = ROOT / cfg['directory']
    if (sha256_file(directory / 'result.json') != cfg['result_sha256']
            or sha256_file(directory.with_name(directory.name + '_supervisor_attempt_1.json')) != cfg['original_terminal_sha256']):
        raise ValueError('Original complete worker and failed observation receipt remain immutable')
    launch = json.loads((directory / 'launch.json').read_text())
    for path, expected in launch['source_sha256'].items():
        source = subprocess.check_output(['git', 'show', cfg['execution_commit'] + ':' + path], cwd=ROOT)
        if hashlib.sha256(source).hexdigest() != expected: raise ValueError('Recorded executed source set, not current-code guess')
    r = json.loads((directory / 'result.json').read_text())
    t = json.loads(directory.with_name(directory.name + '_supervisor_attempt_1.json').read_text())
    if r['live_remote_registration_proof']['sha'] != cfg['execution_commit']:
        raise ValueError('Actual execution commit')
    qualified = qualified_completed_C(r, t, json.loads((ROOT / cfg['resource_limits']).read_text()))
    out = ROOT / cfg['output_receipt']
    if out.exists(): raise ValueError('No receipt overwrite')
    atomic_json(out, {**qualified, 'live_remote_registration_proof': proof,
        'result_sha256': cfg['result_sha256'], 'original_terminal_sha256': cfg['original_terminal_sha256'],
        'executed_source_fingerprints_checked_against_original_commit': True,
        'private_assets_or_GT_read': False})
    print(json.dumps(qualified), flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
