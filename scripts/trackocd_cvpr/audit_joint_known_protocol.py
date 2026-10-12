#!/usr/bin/env python3
"""One owned CPU worker, finite joint protocol proof/solver, no training."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scipy
from src.trackocd_cvpr.joint_known_protocol import joint_feasibility, exhaustive_component_necessary
from src.trackocd_cvpr.known_census_protocol import known_presence
from src.trackocd_cvpr.scoped_train_census import inherited_known_ids, validate_video_scope
from scripts.trackocd_cvpr.audit_known_census_protocol import exact_open_barrier
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration

SOURCES = ('configs/trackocd_cvpr/joint_known_protocol_v1.json',
           'scripts/trackocd_cvpr/audit_joint_known_protocol.py', 'src/trackocd_cvpr/joint_known_protocol.py',
           'src/trackocd_cvpr/known_census_protocol.py', 'src/trackocd_cvpr/disjoint_video_witness.py',
           'scripts/trackocd_cvpr/audit_known_census_protocol.py', 'src/trackocd_cvpr/scoped_train_census.py',
           'src/trackocd_cvpr/metadata_only.py', 'scripts/trackocd_cvpr/audit_train_metadata_containment.py',
           'src/trackocd_v2/io.py', 'configs/trackocd_cvpr/clean_train_gt_scope_approval_v1.json',
           'src/trackocd_cvpr/remote_registration.py',
           'outputs/trackocd_cvpr/cvpr2/clean_train_gt_census_v1/result.json',
           'configs/trackocd_cvpr/autonomous_research.json')


def resources_ok(cfg, safety, root, started, rss):
    mem = {k: int(v.split()[0]) * 1024 for k, v in
           (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    if mem['MemAvailable'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']:
        return 'system_RAM_reserve'
    if rss > cfg['rss_bytes_ceiling']:
        return 'owned_worker_RSS'
    if time.monotonic() - started > cfg['wall_seconds_ceiling']:
        return 'owned_job_wall'
    if shutil.disk_usage(root).free < safety['disk_free_reserve_bytes']:
        return 'free_disk_reserve'
    return None


def supervise(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    receipt = out.with_name(out.name + f'_supervisor_attempt_{args.attempt}.json')
    if out.exists() or receipt.exists():
        raise ValueError('Preserve every previous owned-worker attempt')
    if args.attempt > 1:
        previous = out.with_name(out.name + f'_supervisor_attempt_{args.attempt - 1}.json')
        if not previous.exists() or json.loads(previous.read_text())['successful_owned_execution'] is not False:
            raise ValueError('Only retry a recorded terminal failed pre-output attempt; never observed live work')
    started, peak, reason, peak_threads = time.monotonic(), 0, None, 0
    if resources_ok(cfg, safety, ROOT, started, 0):
        raise RuntimeError('Pre-spawn resource guard')
    child = subprocess.Popen([sys.executable, '-B', '-X', 'pycache_prefix=' + str(out / '_fresh_no_write_cache'),
                              str(Path(__file__).resolve()), '--worker',
                              '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy,
                              '--attempt', str(args.attempt), '--config', args.config], cwd=ROOT)
    try:
        while child.poll() is None:
            status_path = Path(f'/proc/{child.pid}/status')
            rollup_path = Path(f'/proc/{child.pid}/smaps_rollup')
            try:
                status = status_path.read_text()
                measured = process_memory(status, rollup_path.read_text())
                peak = max(peak, measured['guarded_process_RSS_bytes'])
                fields = {line.split(':', 1)[0]: line.split(':', 1)[1] for line in status.splitlines() if ':' in line}
                peak_threads = max(peak_threads, int(fields['Threads'].strip()))
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                if child.poll() is not None:
                    break
                if time.monotonic() - started > 2:
                    reason = 'owned_memory_counter_unavailable'
                    child.kill()
                    break
                time.sleep(cfg['supervisor_poll_seconds'])
                continue
            reason = resources_ok(cfg, safety, ROOT, started, peak)
            if out.exists() and sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > cfg['output_bytes_ceiling']:
                reason = 'owned_output_size'
            if reason:
                if child.poll() is None:
                    child.kill()  # Only the Popen handle created above; no PID/state-file replay.
                break
            time.sleep(cfg['supervisor_poll_seconds'])
        code = child.wait(timeout=10)
    except BaseException:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        raise
    atomic_json(receipt, {'attempt': args.attempt, 'owned_worker_exit_code': code, 'guard_stop_reason': reason,
                         'peak_sampled_owned_worker_RSS_bytes': peak, 'peak_sampled_worker_threads': peak_threads,
                         'wall_seconds_including_imports_network_and_solver': time.monotonic() - started,
                         'only_new_owned_Popen_worker_managed': True, 'training_GPU_hours': 0,
                         'successful_owned_execution': code == 0 and reason is None})
    return 0 if code == 0 and reason is None else 1


def worker(args, cfg, safety):
    if scipy.__version__ != cfg['scipy_version']:
        raise ValueError('Exact existing solver dependency version required')
    if (not sys.dont_write_bytecode or sys.pycache_prefix != str(ROOT / cfg['output_directory'] / '_fresh_no_write_cache')):
        raise ValueError('Fresh isolated no-write bytecode prefix required; do not reuse historical pyc')
    approval = json.loads((ROOT / cfg['approval']).read_text())
    if (approval['external_known_only_Train_census_authorized'] is not True
            or (approval['allowlisted_videos'], approval['excluded_old_heldout_videos']) != (429, 51)):
        raise ValueError('Actual user-authorized derived Known scope required')
    registration = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy,
        allow_public_api=cfg.get('equivalent_fresh_public_ref_API_fallback', False))
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{args.preregistration_commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered source differs')
    out = (ROOT / cfg['output_directory']).resolve()
    if out.exists():
        raise ValueError('No attempt overwrite')
    names = ('private_Known_index', 'private_exclusion_index', 'roles', 'census_result')
    paths = {name: ROOT / cfg[name] for name in names}
    for path in paths.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError('Fixed regular Known input only')
    permitted = {p.resolve() for p in paths.values()} | {(ROOT / name).resolve() for name in SOURCES}
    proc = {Path(p).resolve() for p in ('/proc/meminfo', '/proc/self/status', '/proc/self/smaps_rollup')}
    sys.addaudithook(lambda event, values: exact_open_barrier(event, values, permitted, proc, out))
    started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file())
    def guard():
        nonlocal peak
        measured = process_memory()
        peak = max(peak, measured['guarded_process_RSS_bytes'])
        reason = resources_ok(cfg, safety, ROOT, started, peak)
        output_bytes = sum(p.stat().st_size for p in out.iterdir() if p.is_file()) if out.exists() else 0
        if reason or output_bytes > cfg['output_bytes_ceiling'] or prior_bytes + output_bytes > safety['incremental_storage_hard_bytes']:
            raise RuntimeError('Owned worker resource guard: ' + str(reason))
    guard()
    out.mkdir(parents=True)
    launch = {'preregistration_commit': args.preregistration_commit,
              'source_sha256': {name: sha256_file(ROOT / name) for name in SOURCES},
              'private_input_sha256': {name: cfg[name + '_sha256'] for name in names},
              'scipy_version': scipy.__version__, 'fresh_source_no_existing_bytecode_cache': True,
              'owned_launch_attempt': args.attempt, 'live_remote_registration_proof': registration}
    atomic_json(out / 'launch.json', launch)
    try:
        for name, path in paths.items():
            if sha256_file(path) != cfg[name + '_sha256']:
                raise ValueError('Fixed Known input identity differs')
        if paths['private_Known_index'].stat().st_size != cfg['private_Known_index_bytes']:
            raise ValueError('Known index size differs')
        prior = json.loads(paths['census_result'].read_text())
        if (prior['private_index']['sha256'] != cfg['private_Known_index_sha256']
                or prior['private_exclusion_index_sha256'] != cfg['private_exclusion_index_sha256']
                or prior['roles_source_sha256'] != cfg['roles_sha256']):
            raise ValueError('Actual census binding differs')
        allowed, excluded = validate_video_scope(json.loads(paths['private_exclusion_index'].read_text()))
        known = inherited_known_ids(paths['roles'])
        presence, counts = known_presence(json.loads(paths['private_Known_index'].read_text()), allowed, known)
        if counts != cfg['expected']:
            raise ValueError('Every original census row retained/reconciled')
        guard()
        if cfg.get('exhaustive_component_necessary_only', False):
            public, private = exhaustive_component_necessary(presence)
        else:
            public, private = joint_feasibility(presence, time_limit=cfg['solver_time_limit_seconds'], node_limit=cfg['solver_node_limit'],
                balanced=cfg['four_balanced_final_groups'], union_support=cfg.get('union_support_closure', False),
                compress_patterns=cfg.get('exact_presence_pattern_count_compression', False))
        if public['integer_necessary_conditions']['class_pool'] != cfg['expected_eligible_pool']:
            raise ValueError('Uniformly eligible class pool changed')
        guard()
        for name, path in paths.items():
            if sha256_file(path) != cfg[name + '_sha256']:
                raise ValueError('Known input changed during structural audit')
        private_path = out / 'private_exact_joint_proof_or_assignment.json'
        atomic_json(private_path, {'not_model_facing': True, 'not_a_frozen_protocol': True, **private})
        guard()
        result = {'status': public['status'], **launch, 'structural_result': public,
                  'census_reconciliation': counts, 'private_output': {'bytes': private_path.stat().st_size, 'sha256': sha256_file(private_path)},
                  'interpretation': cfg['interpretation'],
                  'original_annotation_old_target_tables_names_Novel_Val_Test_pixels_features_models_read': False,
                  'optimization_training_steps': 0, 'resources': {'wall_seconds_excluding_network': time.monotonic() - started,
                       **process_memory(), 'peak_guarded_worker_RSS_bytes': peak,
                       'resource_ru_maxrss_bytes_including_preexec_history': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       'training_GPU_hours': 0, 'disk_free_bytes': shutil.disk_usage(ROOT).free}}
        guard()
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'structural_result': public, 'resources': result['resources']}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False})
        raise


def main():
    global SOURCES
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    parser.add_argument('--git-proxy', default='')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--attempt', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--config', choices=('configs/trackocd_cvpr/joint_known_protocol_v1.json',
                                          'configs/trackocd_cvpr/joint_known_protocol_v2_unbalanced.json',
                                          'configs/trackocd_cvpr/joint_known_protocol_v3_unbalanced_ref_transport.json',
                                          'configs/trackocd_cvpr/joint_known_protocol_v4_compressed.json',
                                          'configs/trackocd_cvpr/joint_known_protocol_v5_exhaustive.json'),
                        default='configs/trackocd_cvpr/joint_known_protocol_v1.json')
    args = parser.parse_args()
    SOURCES = (args.config, *SOURCES[1:])
    cfg = json.loads((ROOT / SOURCES[0]).read_text())
    safety = json.loads((ROOT / SOURCES[-1]).read_text())
    if args.worker:
        worker(args, cfg, safety)
        return 0
    return supervise(args, cfg, safety)


if __name__ == '__main__':
    sys.exit(main())
