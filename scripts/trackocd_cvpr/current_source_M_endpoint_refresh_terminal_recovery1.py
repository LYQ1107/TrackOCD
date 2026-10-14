#!/usr/bin/env python3
"""Fresh namespace recovery for one supervisor terminal-counter race.

The prior worker result is evidence of the failure only; this route starts a
new fresh worker and never loads its checkpoint or private history.
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
from src.trackocd_v2.io import atomic_json, sha256_file
from scripts.trackocd_cvpr import current_source_M_endpoint_refresh as base

SCRIPT = 'scripts/trackocd_cvpr/current_source_M_endpoint_refresh_terminal_recovery1.py'
CONFIG = 'configs/trackocd_cvpr/M_current_source_endpoint_refresh_terminal_recovery_v1.json'
NORMAL_CONFIG = 'configs/trackocd_cvpr/M_current_source_endpoint_refresh_v1.json'
FAILED = 'outputs/trackocd_cvpr/cvpr4/M_GT_window_current_source_v1_relative_quarantine_s1027'
RECOVERY_PREFIX = 'outputs/trackocd_cvpr/cvpr4/M_GT_window_current_source_v1_terminal_recovery1'
NORMAL_PREFIX = 'outputs/trackocd_cvpr/cvpr4/M_GT_window_current_source_v1'

base.SOURCES = tuple(dict.fromkeys((*base.SOURCES, SCRIPT, CONFIG,
    CONFIG, FAILED + '/launch.json', FAILED + '/result.json',
    FAILED + '_supervisor_attempt_1.json')))
base.original.SOURCES = base.SOURCES


def configuration(args):
    manifest = json.loads((ROOT / CONFIG).read_text())
    expected = {
        'status', 'base_manifest', 'base_manifest_sha256', 'failed_namespace',
        'failed_launch', 'failed_result', 'failed_terminal', 'failed_launch_sha256',
        'failed_result_sha256', 'failed_terminal_sha256', 'recovery_output_prefix',
        'failure_cause', 'fresh_worker_no_partial_metric_or_checkpoint_reuse',
        'same_current_source_training_protocol_and_resource_guards',
        'no_old_endpoint_or_private_asset_upload', 'formal_M_development_or_model_selection',
    }
    if (set(manifest) != expected or manifest['status'] not in {
            'REGISTERED_CURRENT_SOURCE_ENDPOINT_TERMINAL_RACE_RECOVERY_V1',
            'REGISTERED_CURRENT_SOURCE_ENDPOINT_TRANSPORT_RETRY1'}):
        raise ValueError('Exact current-source terminal recovery manifest required')
    if (manifest['base_manifest'] != NORMAL_CONFIG
            or sha256_file(ROOT / NORMAL_CONFIG) != manifest['base_manifest_sha256']
            or manifest['failed_namespace'] != FAILED
            or manifest['failed_launch'] != FAILED + '/launch.json'
            or manifest['failed_result'] != FAILED + '/result.json'
            or manifest['failed_terminal'] != FAILED + '_supervisor_attempt_1.json'
            or manifest['recovery_output_prefix'] != RECOVERY_PREFIX
            or not manifest['fresh_worker_no_partial_metric_or_checkpoint_reuse']
            or not manifest['same_current_source_training_protocol_and_resource_guards']
            or not manifest['no_old_endpoint_or_private_asset_upload']
            or manifest['formal_M_development_or_model_selection']):
        raise ValueError('Only the registered fresh terminal recovery is allowed')
    for path, digest in (
            (manifest['failed_launch'], manifest['failed_launch_sha256']),
            (manifest['failed_result'], manifest['failed_result_sha256']),
            (manifest['failed_terminal'], manifest['failed_terminal_sha256'])):
        if sha256_file(ROOT / path) != digest:
            raise ValueError('Preserved failed endpoint evidence changed: ' + path)
    terminal = json.loads((ROOT / manifest['failed_terminal']).read_text())
    result = json.loads((ROOT / manifest['failed_result']).read_text())
    if (terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] != 'owned_memory_counter_unavailable'
            or not terminal['complete_result_present']
            or terminal['successful_owned_execution']
            or result['status'] != 'COMPLETED_FIXED_M_TRIAL_NOT_DEV_PRED_JOINT_OR_GOAL_ACCEPTANCE'
            or result['summary']['optimizer_steps'] != 200
            or result['old_R_C_M_or_pilot_warmstart']
            or result['dev_final_Val_Test_or_new_nonKnown_labels_used']):
        raise ValueError('Expected only the preserved terminal-counter failure')
    old_prefix = base.OUTPUT_PREFIX
    base.OUTPUT_PREFIX = NORMAL_PREFIX
    try:
        cfg = base.configuration(args)
    finally:
        base.OUTPUT_PREFIX = old_prefix
    cfg['output_prefix'] = RECOVERY_PREFIX
    cfg['output_directory'] = RECOVERY_PREFIX + '_' + args.variant + '_s' + str(args.seed)
    return cfg


def worker(args, cfg, safety):
    # base.worker creates a fresh endpoint from cfg; it never consumes FAILED.
    return base.worker(args, cfg, safety)


def worker_command(args, out):
    return [sys.executable, '-B', '-X', 'pycache_prefix=' + str(out / '_fresh_no_write_cache'),
        str(ROOT / SCRIPT), '--worker', '--variant', args.variant, '--seed', str(args.seed),
        '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy]


def supervise(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    terminal = out.with_name(out.name + '_supervisor_attempt_1.json')
    if out.exists() or terminal.exists():
        raise ValueError('Preserve every previous terminal-recovery namespace')
    started, peak, threads, reason = time.monotonic(), 0, 0, None
    if base.original.resources_ok(cfg, safety, ROOT, started, 0):
        raise RuntimeError('Pre-spawn terminal-recovery resource guard')
    child = subprocess.Popen(worker_command(args, out), cwd=ROOT)
    try:
        while child.poll() is None:
            status_path = Path(f'/proc/{child.pid}/status')
            try:
                status = status_path.read_text()
                rollup = Path(f'/proc/{child.pid}/smaps_rollup').read_text()
                peak = max(peak, base.original.process_memory(status, rollup)['guarded_process_RSS_bytes'])
                fields = {x.split(':', 1)[0]: x.split(':', 1)[1] for x in status.splitlines() if ':' in x}
                threads = max(threads, int(fields['Threads'].strip()))
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                if child.poll() is not None:
                    break
                try:
                    kernel_terminal = base.original.kernel_terminal_status(status_path.read_text())
                except (FileNotFoundError, ProcessLookupError):
                    kernel_terminal = False
                if kernel_terminal:
                    child.wait(timeout=1)
                    break
                if child.poll() is not None:
                    break
                if base.original.settled_after_counter_error(child):
                    break
                if time.monotonic() - started > 2:
                    reason = 'owned_memory_counter_unavailable'; child.kill(); break
                time.sleep(.2); continue
            reason = base.original.resources_ok(cfg, safety, ROOT, started, peak)
            if out.exists() and sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > cfg['output_bytes_ceiling']:
                reason = 'owned_output_size'
            if reason:
                if child.poll() is None: child.kill()
                break
            time.sleep(.2)
        code = child.wait(timeout=10)
    except BaseException:
        if child.poll() is None: child.kill()
        child.wait(timeout=10); raise
    progress = json.loads((out / 'progress.json').read_text()) if (out / 'progress.json').exists() else {}
    complete = (out / 'result.json').is_file()
    atomic_json(terminal, {
        'owned_worker_exit_code': code, 'guard_stop_reason': reason,
        'successful_owned_execution': code == 0 and reason is None and complete,
        'complete_result_present': complete, 'terminal_state_check_repair': True,
        'only_new_owned_Popen_worker_managed': True,
        'last_persisted_completed_optimizer_steps': progress.get('step', 0),
        'peak_sampled_owned_worker_RSS_bytes': peak, 'peak_sampled_worker_threads': threads,
        'wall_seconds_including_imports_and_network': time.monotonic() - started,
        'training_GPU_hours': 0, 'execution_device': 'CPU'})
    return 0 if code == 0 and reason is None and complete else 1


def launch(args, cfg, safety):
    proof, _ = base.source_registration(args)
    out = ROOT / cfg['output_directory']
    terminal = out.with_name(out.name + '_supervisor_attempt_1.json')
    receipt = out.with_name(out.name + '_persistent_launch.json')
    log = out.with_name(out.name + '_persistent_stdout.log')
    if any(path.exists() for path in (out, terminal, receipt, log)):
        raise ValueError('Never relaunch terminal-recovery namespace')
    marker = os.fsencode(str(ROOT / SCRIPT)) + b'\0'
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try: content = path.read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        if marker in content and (b'--worker\0' in content or b'--supervisor\0' in content):
            raise ValueError('Terminal recovery is already live')
    if base.original.resources_ok(cfg, safety, ROOT, time.monotonic(), 0):
        raise RuntimeError('Pre-spawn terminal-recovery resource guard')
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-B', str(ROOT / SCRIPT), '--supervisor', '--variant', args.variant,
        '--seed', str(args.seed), '--preregistration-commit', args.preregistration_commit,
        '--git-proxy', args.git_proxy]
    with log.open('xb', buffering=0) as stream:
        os.chmod(log, 0o600)
        child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
    try: ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError: ticks = None
    atomic_json(receipt, {
        'status': 'OWNED_CURRENT_SOURCE_M_ENDPOINT_TERMINAL_RECOVERY_LAUNCHED_NOT_COMPLETED',
        'supervisor_pid': child.pid, 'linux_start_ticks': ticks,
        'command_sha256': hashlib.sha256(b'\0'.join(os.fsencode(x) for x in cmd)).hexdigest(),
        'output_namespace': cfg['output_directory'], 'live_remote_registration_proof': proof,
        'start_new_session': True, 'normal_terminal_receipt': str(terminal.relative_to(ROOT)),
        'preserved_failed_namespace': FAILED, 'fresh_worker_no_old_checkpoint_or_history_reuse': True,
        'training_GPU_hours': 0})
    print(json.dumps({'status': 'LAUNCHED_NOT_COMPLETED', 'receipt': str(receipt.relative_to(ROOT))}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True); parser.add_argument('--git-proxy', default='')
    parser.add_argument('--variant', choices=('relative_quarantine', 'relative_weighted', 'pooled_weighted'), required=True)
    parser.add_argument('--seed', type=int, choices=(1027, 1028, 1029), required=True)
    mode = parser.add_mutually_exclusive_group(); mode.add_argument('--worker', action='store_true'); mode.add_argument('--supervisor', action='store_true')
    args = parser.parse_args(); cfg = configuration(args)
    if args.worker:
        _, safety = base.source_registration(args)
        return worker(args, cfg, safety)
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.supervisor: return supervise(args, cfg, safety)
    return launch(args, cfg, safety)


if __name__ == '__main__': sys.exit(main())
