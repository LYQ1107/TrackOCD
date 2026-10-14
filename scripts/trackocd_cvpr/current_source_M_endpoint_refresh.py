#!/usr/bin/env python3
"""Fresh fixed-M endpoint refresh bound to the current source closure.

This wrapper reuses the frozen 200-step worker implementation but supplies a
new output namespace and an explicit source-refresh manifest.  It never loads
an old endpoint and it does not start development evaluation automatically.
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
from src.trackocd_cvpr.remote_registration import verify_registration
from src.trackocd_cvpr.M_endpoint_source_compatibility import CURRENT_SOURCES
from scripts.trackocd_cvpr import pilot_memory_optimizer as original

SCRIPT = 'scripts/trackocd_cvpr/current_source_M_endpoint_refresh.py'
CONFIG = 'configs/trackocd_cvpr/M_current_source_endpoint_refresh_v1.json'
BASE_CONFIG = 'configs/trackocd_cvpr/memory_optimizer_pilot_v1.json'
SCIENTIFIC_CONFIG = 'configs/trackocd_cvpr/M_GT_scientific_window_v1.json'
OUTPUT_PREFIX = 'outputs/trackocd_cvpr/cvpr4/M_GT_window_current_source_v1'
SOURCES = tuple(dict.fromkeys((*original.SOURCES, SCRIPT, CONFIG, SCIENTIFIC_CONFIG,
    'src/trackocd_cvpr/M_endpoint_source_compatibility.py',
    'src/trackocd_cvpr/memory_scientific_training.py',
    'src/trackocd_cvpr/version_lineage_audit.py',
    'outputs/trackocd_cvpr/cvpr4/memory_optimizer_pilot_v1/result.json',
    'outputs/trackocd_cvpr/cvpr4/memory_optimizer_pilot_v1_supervisor_attempt_1.json')))
original.SOURCES = SOURCES


def configuration(args):
    manifest = json.loads((ROOT / CONFIG).read_text())
    expected = {
        'status', 'base_optimizer_config', 'base_optimizer_config_sha256',
        'scientific_window_config', 'scientific_window_config_sha256', 'output_prefix',
        'variants', 'training_seeds', 'fixed_optimizer_steps', 'trainable_parameters',
        'execution_device', 'CPU_compute_workers', 'rss_bytes_ceiling',
        'wall_seconds_ceiling', 'output_bytes_ceiling',
        'models_orders_inputs_loss_and_guards_unchanged',
        'fresh_initialization_no_old_endpoint_warmstart',
        'partial_metric_reuse_or_old_namespace_overwrite',
        'model_selection_or_dev_final_selection',
        'final_Val_Test_or_new_nonKnown_semantics',
        'automatic_full_M_development_or_Goal_acceptance', 'training_GPU_hours',
        'current_source_paths', 'source_provenance_target',
        'no_private_data_weights_or_prediction_history_upload',
    }
    if set(manifest) != expected:
        raise ValueError('Exact current-source endpoint refresh manifest required')
    if (manifest['status'] != 'REGISTERED_CURRENT_SOURCE_FIXED_M_ENDPOINT_REFRESH_PENDING'
            or manifest['base_optimizer_config'] != BASE_CONFIG
            or manifest['scientific_window_config'] != SCIENTIFIC_CONFIG
            or manifest['output_prefix'] != OUTPUT_PREFIX
            or manifest['variants'] != {
                'relative_quarantine': {'pooled_control': False, 'quarantine': True},
                'relative_weighted': {'pooled_control': False, 'quarantine': False},
                'pooled_weighted': {'pooled_control': True, 'quarantine': False}}
            or manifest['training_seeds'] != [1027, 1028, 1029]
            or manifest['fixed_optimizer_steps'] != 200
            or manifest['trainable_parameters'] != 161
            or manifest['execution_device'] != 'CPU'
            or manifest['CPU_compute_workers'] != 1
            or manifest['rss_bytes_ceiling'] != 3221225472
            or manifest['wall_seconds_ceiling'] != 3000
            or manifest['output_bytes_ceiling'] != 33554432
            or not manifest['models_orders_inputs_loss_and_guards_unchanged']
            or not manifest['fresh_initialization_no_old_endpoint_warmstart']
            or manifest['partial_metric_reuse_or_old_namespace_overwrite']
            or manifest['model_selection_or_dev_final_selection']
            or manifest['final_Val_Test_or_new_nonKnown_semantics']
            or manifest['automatic_full_M_development_or_Goal_acceptance']
            or manifest['training_GPU_hours'] != 0
            or not manifest['no_private_data_weights_or_prediction_history_upload']):
        raise ValueError('Only the frozen fresh 200-step endpoint refresh is allowed')
    if (sha256_file(ROOT / BASE_CONFIG) != manifest['base_optimizer_config_sha256']
            or sha256_file(ROOT / SCIENTIFIC_CONFIG) != manifest['scientific_window_config_sha256']
            or manifest['current_source_paths'] != CURRENT_SOURCES):
        raise ValueError('Current source/config hashes changed after registration')
    if any(sha256_file(ROOT / path) != digest for path, digest in CURRENT_SOURCES.items()):
        raise ValueError('Current source closure does not match the refresh manifest')
    if args.variant not in manifest['variants'] or args.seed not in manifest['training_seeds']:
        raise ValueError('Explicit registered variant and independent seed required')
    base = json.loads((ROOT / BASE_CONFIG).read_text())
    window = json.loads((ROOT / SCIENTIFIC_CONFIG).read_text())
    cfg = {**base, **window, **window['training'], **window['proposed_execution'],
        'scientific_M_trial': True, 'seed': args.seed, 'variant': args.variant,
        'output_prefix': OUTPUT_PREFIX,
        'output_directory': OUTPUT_PREFIX + '_' + args.variant + '_s' + str(args.seed)}
    return cfg


def source_registration(args):
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit,
                                args.git_proxy, allow_public_api=True)
    for path in SOURCES:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Actual current-source refresh dependency is not registered: ' + path)
    return proof, safety


def worker(args, cfg, safety):
    # The imported worker consumes cfg and its module-level source closure; it
    # does not consult the old scientific output prefix once cfg is supplied.
    original.worker(args, cfg, safety)
    return 0


def worker_command(args, out):
    return [sys.executable, '-B', '-X', 'pycache_prefix=' + str(out / '_fresh_no_write_cache'),
        str(ROOT / SCRIPT), '--worker', '--variant', args.variant, '--seed', str(args.seed),
        '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy]


def supervise(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    terminal = out.with_name(out.name + '_supervisor_attempt_1.json')
    if out.exists() or terminal.exists():
        raise ValueError('Preserve every previous current-source endpoint attempt')
    started, peak, threads, reason = time.monotonic(), 0, 0, None
    if original.resources_ok(cfg, safety, ROOT, started, 0):
        raise RuntimeError('Pre-spawn current-source refresh resource guard')
    child = subprocess.Popen(worker_command(args, out), cwd=ROOT)
    try:
        while child.poll() is None:
            status_path = Path(f'/proc/{child.pid}/status')
            try:
                status = status_path.read_text()
                rollup = Path(f'/proc/{child.pid}/smaps_rollup').read_text()
                peak = max(peak, original.process_memory(status, rollup)['guarded_process_RSS_bytes'])
                fields = {x.split(':', 1)[0]: x.split(':', 1)[1] for x in status.splitlines() if ':' in x}
                threads = max(threads, int(fields['Threads'].strip()))
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                if child.poll() is not None or original.settled_after_counter_error(child):
                    break
                if time.monotonic() - started > 2:
                    reason = 'owned_memory_counter_unavailable'; child.kill(); break
                time.sleep(.2); continue
            reason = original.resources_ok(cfg, safety, ROOT, started, peak)
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
        'complete_result_present': complete, 'only_new_owned_Popen_worker_managed': True,
        'last_persisted_completed_optimizer_steps': progress.get('step', 0),
        'peak_sampled_owned_worker_RSS_bytes': peak, 'peak_sampled_worker_threads': threads,
        'wall_seconds_including_imports_and_network': time.monotonic() - started,
        'training_GPU_hours': 0, 'execution_device': 'CPU'})
    return 0 if code == 0 and reason is None and complete else 1


def launch(args, cfg, safety):
    proof, _ = source_registration(args)
    out = ROOT / cfg['output_directory']
    terminal = out.with_name(out.name + '_supervisor_attempt_1.json')
    receipt = out.with_name(out.name + '_persistent_launch.json')
    log = out.with_name(out.name + '_persistent_stdout.log')
    if any(path.exists() for path in (out, terminal, receipt, log)):
        raise ValueError('Never relaunch a current-source endpoint namespace')
    marker = os.fsencode(str(ROOT / SCRIPT)) + b'\0'
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try: content = path.read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        if marker in content and (b'--worker\0' in content or b'--supervisor\0' in content):
            raise ValueError('Current-source endpoint refresh is already live')
    if original.resources_ok(cfg, safety, ROOT, time.monotonic(), 0):
        raise RuntimeError('Pre-spawn current-source refresh resource guard')
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
        'status': 'OWNED_CURRENT_SOURCE_M_ENDPOINT_SUPERVISOR_LAUNCHED_NOT_COMPLETED',
        'supervisor_pid': child.pid, 'linux_start_ticks': ticks,
        'command_sha256': hashlib.sha256(b'\0'.join(os.fsencode(x) for x in cmd)).hexdigest(),
        'output_namespace': cfg['output_directory'], 'live_remote_registration_proof': proof,
        'start_new_session': True,
        'normal_terminal_receipt': str(terminal.relative_to(ROOT)),
        'no_old_endpoint_or_private_asset_upload': True, 'training_GPU_hours': 0})
    print(json.dumps({'status': 'LAUNCHED_NOT_COMPLETED', 'receipt': str(receipt.relative_to(ROOT))}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True); parser.add_argument('--git-proxy', default='')
    parser.add_argument('--variant', choices=('relative_quarantine', 'relative_weighted', 'pooled_weighted'), required=True)
    parser.add_argument('--seed', type=int, choices=(1027, 1028, 1029), required=True)
    mode = parser.add_mutually_exclusive_group(); mode.add_argument('--worker', action='store_true'); mode.add_argument('--supervisor', action='store_true')
    args = parser.parse_args(); cfg = configuration(args)
    _, safety = source_registration(args) if args.worker else (None, json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text()))
    if args.worker: return worker(args, cfg, safety)
    if args.supervisor: return supervise(args, cfg, safety)
    return launch(args, cfg, safety)


if __name__ == '__main__': sys.exit(main())
