#!/usr/bin/env python3
"""Detach ONE gated full-dev supervisor with exact boot/start identity."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json
from scripts.trackocd_cvpr.evaluate_raw_membership_development import (
    parser, configuration, source_gate, busy, runtime, SCRIPT, explicit_args)


def supervisor_command(args):
    return [sys.executable, '-B', str(ROOT/SCRIPT), '--preregistration-commit', args.preregistration_commit,
            '--git-proxy', args.git_proxy, *explicit_args(args)]


def main():
    args = parser().parse_args()
    if args.worker: raise ValueError('Launcher never performs private worker evaluation')
    cfg = configuration(ROOT, args.control, args.seed)
    safety = json.loads((ROOT/'configs/trackocd_cvpr/autonomous_research.json').read_text())
    proof = source_gate(args, cfg, safety)
    out = ROOT/cfg['output_directory']; receipt = out.with_name(out.name+'_persistent_launch.json')
    log = out.with_name(out.name+'_persistent_stdout.log'); terminal = out.with_name(out.name+'_supervisor_attempt_1.json')
    if any(p.exists() for p in (out, receipt, log, terminal)): raise ValueError('Preserve all previous starts/namespaces')
    if busy(): raise RuntimeError('Observe live owned research process; no duplicate')
    start = time.monotonic()
    mem = {k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}
    if (runtime.resources_ok(cfg, safety, ROOT, start, 0)
            or mem['MemAvailable'] < safety['system_ram_reserve_fraction']*mem['MemTotal']+cfg['rss_bytes_ceiling']):
        raise RuntimeError('Original reserves and planned RSS before detached spawn')
    command = supervisor_command(args); out.parent.mkdir(parents=True, exist_ok=True)
    with log.open('xb', buffering=0) as stream:
        os.chmod(log, 0o600)
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, close_fds=True, start_new_session=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
    try: ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError: ticks = None
    atomic_json(receipt, {'status':'OWNED_SUPERVISOR_LAUNCHED_NOT_COMPLETED', 'supervisor_pid':child.pid,
        'linux_start_ticks':ticks, 'linux_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'command_sha256':hashlib.sha256(b'\0'.join(os.fsencode(s) for s in command)).hexdigest(),
        'live_remote_registration_proof':proof, 'output_namespace':cfg['output_directory'],
        'start_new_session':True, 'normal_terminal_receipt':str(terminal.relative_to(ROOT)),
        'no_private_assets_or_foreign_process_changes':True, 'training_GPU_hours':0})
    print(json.dumps({'supervisor_pid':child.pid, 'receipt':str(receipt.relative_to(ROOT)), 'status':'LAUNCHED_NOT_SUCCESS_RECEIPT'}), flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
