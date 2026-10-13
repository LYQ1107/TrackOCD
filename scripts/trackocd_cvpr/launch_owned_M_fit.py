#!/usr/bin/env python3
"""Detach only an owned guarded fixed-M supervisor; no direct asset loading."""
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
from scripts.trackocd_cvpr.pilot_memory_optimizer import SCIENTIFIC_CONFIG, CONFIG
from src.trackocd_cvpr.memory_transport_recovery import recovery_configuration


def fixed_configuration(args):
    base = json.loads((ROOT / CONFIG).read_text()); window = json.loads((ROOT / SCIENTIFIC_CONFIG).read_text())
    cfg = {**base, **window, **window['training'], **window['proposed_execution'],
        'scientific_M_trial': True, 'seed': args.seed, 'variant': args.variant,
        'output_directory': f'outputs/trackocd_cvpr/cvpr4/M_GT_window_v1_{args.variant}_s{args.seed}'}
    return recovery_configuration(ROOT, cfg, args.recovery_attempt)


def supervisor_command(args):
    cmd = [sys.executable, '-B', str(ROOT / 'scripts/trackocd_cvpr/pilot_memory_optimizer.py'),
        '--config', SCIENTIFIC_CONFIG, '--variant', args.variant, '--seed', str(args.seed),
        '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy]
    if args.recovery_attempt is not None: cmd += ['--recovery-attempt', args.recovery_attempt]
    return cmd


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit', required=True); p.add_argument('--git-proxy', default='')
    p.add_argument('--variant', choices=('relative_quarantine', 'relative_weighted', 'pooled_weighted'), required=True)
    p.add_argument('--seed', type=int, choices=(1027, 1028, 1029), required=True)
    p.add_argument('--recovery-attempt', choices=('transport_retry1', 'terminal_replica1'))
    args = p.parse_args()
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    cfg, recovery_sources = fixed_configuration(args)
    for path in ('scripts/trackocd_cvpr/launch_owned_M_fit.py', 'scripts/trackocd_cvpr/pilot_memory_optimizer.py',
                 'src/trackocd_cvpr/memory_transport_recovery.py', CONFIG, SCIENTIFIC_CONFIG, *recovery_sources):
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Actual M launcher/transport source must be registered')
    out = ROOT / cfg['output_directory']; receipt = out.with_name(out.name + '_persistent_launch.json')
    log = out.with_name(out.name + '_persistent_stdout.log'); terminal = out.with_name(out.name + '_supervisor_attempt_1.json')
    if any(x.exists() for x in (out, receipt, log, terminal)): raise ValueError('Preserve every previous namespace')
    marker = os.fsencode(str(ROOT / 'scripts/trackocd_cvpr/pilot_memory_optimizer.py')) + b'\0'
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try: value = path.read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        if marker in value: raise ValueError('A M supervisor/worker is live; observe it, do not duplicate')
    if resources_ok(cfg, safety, ROOT, time.monotonic(), 0): raise RuntimeError('Pre-spawn resource guard')
    out.parent.mkdir(parents=True, exist_ok=True); cmd = supervisor_command(args)
    with log.open('xb', buffering=0) as stream:
        os.chmod(log, 0o600)
        child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
    try: ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError: ticks = None
    atomic_json(receipt, {'status': 'OWNED_FIXED_M_SUPERVISOR_LAUNCHED_NOT_COMPLETED',
        'supervisor_pid': child.pid, 'linux_start_ticks': ticks,
        'command_sha256': hashlib.sha256(b'\0'.join(os.fsencode(x) for x in cmd)).hexdigest(),
        'output_namespace': cfg['output_directory'], 'live_remote_registration_proof': proof,
        'start_new_session': True, 'normal_terminal_receipt': str(terminal.relative_to(ROOT)),
        'no_private_GT_cache_or_weight_loading_by_launcher': True, 'no_foreign_process_changes': True})
    print(json.dumps({'status': 'LAUNCHED_NOT_COMPLETED', 'receipt': str(receipt.relative_to(ROOT))}), flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
