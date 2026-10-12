#!/usr/bin/env python3
"""Detach ONLY the owned guarded C supervisor from a short tool session.

No private assets read, no broad process management, no external daemon/service
change. Worker retains original guards, fresh registration and source checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json
from src.trackocd_cvpr.remote_registration import verify_registration
from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok
from scripts.trackocd_cvpr.evaluate_C_development import CONFIG, MODELS, VARIANTS, recovery_configuration


def supervisor_command(args):
    command = [sys.executable, '-B', str(ROOT / 'scripts/trackocd_cvpr/evaluate_C_development.py'),
        '--model', args.model, '--seed', str(args.seed),
        '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy]
    if args.recovery_attempt is not None:
        command += ['--recovery-attempt', args.recovery_attempt]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    parser.add_argument('--git-proxy', default='')
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--seed', choices=(1027, 1028, 1029), type=int, required=True)
    parser.add_argument('--recovery-attempt', choices=('transport_retry1',))
    args = parser.parse_args()
    if args.model.startswith('Raw_') and args.seed != 1027:
        raise ValueError('Raw is not three independent trained seeds')
    cfg, recovery_sources = recovery_configuration(ROOT, json.loads((ROOT / CONFIG).read_text()), args)
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for source in ('scripts/trackocd_cvpr/launch_owned_C_development.py',
                   'scripts/trackocd_cvpr/evaluate_C_development.py',
                   'scripts/trackocd_cvpr/qualify_policy_training_input.py', CONFIG, *recovery_sources):
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + source], cwd=ROOT) != (ROOT / source).read_bytes():
            raise ValueError('Persistent launcher/source evidence must be registered')
    name = cfg['output_prefix'] + '_' + args.model + ('_s' + str(args.seed) if args.model in VARIANTS else '')
    out = ROOT / name
    receipt = out.with_name(out.name + '_persistent_launch.json')
    log = out.with_name(out.name + '_persistent_stdout.log')
    terminal = out.with_name(out.name + '_supervisor_attempt_1.json')
    if any(p.exists() for p in (out, receipt, log, terminal)):
        raise ValueError('Never overwrite/relaunch any prior attempt')
    # Diagnostic scan, not kill/PID reuse: reject a second C evaluator.
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try: content = path.read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        if b'/scripts/trackocd_cvpr/evaluate_C_development.py\x00' in content:
            raise ValueError('A C evaluator already exists; observe it, do not duplicate')
    if resources_ok(cfg, safety, ROOT, time.monotonic(), 0):
        raise RuntimeError('Pre-spawn resource guard')
    command = supervisor_command(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    with log.open('xb', buffering=0) as stream:
        os.chmod(log, 0o600)
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
    # PID alone is NOT identity. Monitoring uses command + Linux start ticks.
    try:
        start_ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        start_ticks = None
    atomic_json(receipt, {'status': 'OWNED_SUPERVISOR_LAUNCHED_NOT_COMPLETED',
        'supervisor_pid': child.pid, 'linux_start_ticks': start_ticks,
        'command_sha256': hashlib.sha256(b'\0'.join(os.fsencode(s) for s in command)).hexdigest(),
        'live_remote_registration_proof': proof, 'output_namespace': name,
        'start_new_session': True, 'normal_terminal_receipt': str(terminal.relative_to(ROOT)),
        'no_private_assets_or_foreign_process_changes': True, 'training_GPU_hours': 0})
    print(json.dumps({'supervisor_pid': child.pid, 'receipt': str(receipt.relative_to(ROOT)),
                      'status': 'LAUNCHED_NOT_A_SUCCESS_RECEIPT'}), flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
