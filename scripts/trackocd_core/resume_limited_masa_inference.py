#!/usr/bin/env python3
"""Recover scheduling without replacing frozen executors or live owned workers.

Adopt only exact current-host process/assignment identities, never historical
PIDs. An orphan's exit status is unknowable: require its atomic done receipt
and every immutable case, and report this distinction explicitly.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file


def snapshot(pid, proc=Path('/proc')):
    try:
        directory = proc / str(pid)
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'uid': directory.stat().st_uid,
                'start_ticks': fields[19], 'state': fields[0],
                'argv': (directory / 'cmdline').read_bytes().decode().strip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def same_process(witness, proc=Path('/proc')):
    current = snapshot(witness['pid'], proc)
    return (current is not None and current['state'] != 'Z'
            and all(current[k] == witness[k] for k in ('pid', 'uid', 'start_ticks', 'argv')))


def headroom(witnesses, peak, proc=Path('/proc')):
    total = 0
    for witness in witnesses:
        if not same_process(witness, proc):
            continue
        try:
            rss = int((proc / str(witness['pid']) / 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except FileNotFoundError:
            continue
        total += max(0, peak - rss)
    return total


def verify_group(out, plan, group, identity, attempt, completed_case):
    ids = plan['groups'][group]
    receipt = json.loads((out / f'{group}_{attempt}_done.json').read_text())
    if receipt['group'] != group or len(receipt['completed']) != len(ids) or set(receipt['completed']) != set(ids):
        raise ValueError('Atomic group terminal receipt does not cover every registered case')
    if any(completed_case(out, plan['jobs'][key], identity) is None for key in ids):
        raise ValueError('Every adopted terminal case must actually be sealed')
    return receipt


def main(commit):
    from scripts.trackocd_core import run_limited_masa_inference as frozen
    from src.trackocd_core.limited_masa_plan import make_plan
    from src.trackocd_core.limited_masa_features import completed_feature
    cfg, fc, manifest, identity = frozen.inputs()
    out = ROOT / cfg['output_directory']
    sources = (*frozen.SOURCES, str(Path(__file__).resolve().relative_to(ROOT)))
    remote = subprocess.check_output(['git', '-c', 'http.proxy=http://127.0.0.1:17890', 'ls-remote', 'origin', 'refs/heads/codex/trackocd-core-training-limited-masa'], text=True, timeout=25).split()[0]
    if remote != commit:
        raise ValueError('Exact remote registration required for recovery source')
    for name in sources:
        if subprocess.check_output(['git', 'show', f'{commit}:{name}']) != (ROOT / name).read_bytes():
            raise ValueError('Recovery/unchanged scientific source differs from registered commit')
    if (out / 'full_inference_manifest.json').exists():
        raise ValueError('Preserve already completed full inference')
    plan_path = out / 'plan.json'
    plan = json.loads(plan_path.read_text())
    freeze = json.loads((ROOT / fc['model_freeze']).read_text())
    policy = json.loads((ROOT / 'outputs/trackocd_core/core_training/train_first_v1/policy/training_receipt.json').read_text())
    if plan != make_plan(cfg, freeze, policy, [r['video_id'] for r in manifest['records']]):
        raise ValueError('Never alter the prospective 840-execution plan')
    plan_sha = sha256_file(plan_path)
    script = str((ROOT / 'scripts/trackocd_core/run_limited_masa_inference.py').resolve())
    own_script = str(Path(__file__).resolve())
    active, adopted = {}, []
    assignments = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdecimal() or int(directory.name) == os.getpid():
            continue
        witness = snapshot(directory.name)
        if not witness or witness['uid'] != os.getuid() or witness['state'] == 'Z':
            continue
        argv = witness['argv']
        if len(argv) < 2 or argv[1] not in (script, own_script):
            continue
        if '--worker-group' not in argv:
            raise RuntimeError('Another live inference scheduler exists; do not duplicate it')
        if len(argv) != 6 or argv[2] != '--worker-group' or argv[4] != '--assignment':
            raise ValueError('Unrecognized live worker invocation; no interference')
        group = argv[3]
        path = Path(argv[5])
        if path.parent != out or path.is_symlink() or not path.name.startswith('assignment_'):
            raise ValueError('Unrecognized assignment location')
        assignment = json.loads(path.read_text())
        if assignment['identity'] != identity or assignment['plan_sha256'] != plan_sha:
            raise ValueError('Cannot adopt a worker with different immutable inputs')
        subprocess.check_call(['git', 'merge-base', '--is-ancestor', assignment['preregistration_commit'], commit])
        for name in frozen.SOURCES:
            if subprocess.check_output(['git', 'show', f"{assignment['preregistration_commit']}:{name}"]) != (ROOT / name).read_bytes():
                raise ValueError('Original worker scientific source differs')
        environment = (directory / 'environ').read_bytes().split(b'\0')
        gpu = next(v.split(b'=', 1)[1].decode() for v in environment if v.startswith(b'CUDA_VISIBLE_DEVICES='))
        if not gpu.startswith('GPU-') or group not in plan['groups'] or group in active:
            raise ValueError('Unique registered group and actual UUID required')
        witness.update(group=group, GPU_UUID=gpu, assignment=str(path), attempt=assignment['attempt'])
        active[group] = {'witness': witness, 'child': None, 'attempt': assignment['attempt'], 'gpu': gpu}
        adopted.append(witness)
        assignments[str(path)] = assignment
    if len({v['gpu'] for v in active.values()}) != len(active):
        raise ValueError('Multiple existing workers on one GPU; defer without intervention')
    if len(active) > cfg['max_concurrent_workers']:
        raise ValueError('Existing workers exceed registered concurrency; no interference')
    for record in manifest['records']:
        if completed_feature(ROOT / fc['output_directory'] / 'shards', record['video_id'], manifest['config_sha256'], record['input_npz_sha256']) != record:
            raise ValueError('Actual atomic feature identity differs')
    elapsed_before = max((time.time() - Path(p).stat().st_mtime for p in assignments), default=0)
    started = time.monotonic() - elapsed_before
    attempt = uuid.uuid4().hex
    assignment_path = out / f'assignment_{attempt}.json'
    atomic_json(assignment_path, {'attempt': attempt, 'preregistration_commit': commit, 'identity': identity, 'plan_sha256': plan_sha})
    already = []
    for group, ids in plan['groups'].items():
        if group not in active and all(frozen.completed_case(out, plan['jobs'][key], identity) is not None for key in ids):
            already.append(group)
    pending = sorted(set(plan['groups']) - set(active) - set(already), key=lambda g: (-len(plan['groups'][g]), g))
    done, children, logs, adopted_completed, error = list(already), [], [], [], None
    atomic_json(out / f'recovery_{attempt}_started.json', {'preregistration_commit': commit, 'source_sha256': {n: sha256_file(ROOT / n) for n in sources}, 'identity': identity,
        'plan_sha256': plan_sha, 'adopted_current_host_workers': adopted, 'prior_assignments': assignments,
        'elapsed_stage_seconds_before_resume': elapsed_before, 'already_completed_groups_not_rerun': already,
        'models_checkpoints_operating_points_plan_or_frozen_executors_modified': False})
    try:
        while active or pending:
            frozen.guard(cfg, started)
            for group, entry in list(active.items()):
                child = entry['child']
                ended = child.poll() is not None if child else not same_process(entry['witness'])
                if not ended:
                    continue
                if child and child.returncode != 0:
                    raise RuntimeError(f'Owned group failed: {group}; no automatic retry')
                verify_group(out, plan, group, identity, entry['attempt'], frozen.completed_case)
                if child is None:
                    adopted_completed.append({'group': group, 'original_attempt': entry['attempt'], 'orphan_exit_status': 'UNAVAILABLE_NOT_A_CHILD', 'atomic_terminal_and_every_seal_verified': True})
                done.append(group)
                del active[group]
            witnesses = [entry['witness'] for entry in active.values()]
            remaining = headroom(witnesses, cfg['host_planned_peak_per_worker_bytes'])
            if pending and len(active) < cfg['max_concurrent_workers']:
                m = frozen.ram()
                evaluator_remaining = frozen.counterpart_reservation(out, 'evaluator_progress.json')
                if m['MemAvailable'] - remaining - evaluator_remaining - cfg['host_planned_peak_per_worker_bytes'] >= m['MemTotal'] * cfg['system_RAM_reserve_fraction']:
                    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10)
                    choices = []
                    occupied = {entry['gpu'] for entry in active.values()}
                    for line in rows.splitlines():
                        gpu, free, utilization = [v.strip() for v in line.split(',')]
                        if gpu not in occupied and int(free) * 2**20 >= cfg['GPU_planned_peak_bytes'] + cfg['GPU_free_reserve_bytes']:
                            choices.append((int(utilization) == 0, int(free), gpu))
                    if choices:
                        gpu = max(choices)[2]
                        group = pending.pop(0)
                        log = (out / f'{group}_{attempt}.log').open('x')
                        logs.append(log)
                        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
                        child = subprocess.Popen([sys.executable, script, '--worker-group', group, '--assignment', str(assignment_path)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        children.append(child)
                        witness = snapshot(child.pid)
                        if witness is None:
                            raise RuntimeError('Fresh owned worker exited before identity witness')
                        active[group] = {'witness': witness, 'child': child, 'attempt': attempt, 'gpu': gpu}
                        print(json.dumps({'launched_group': group, 'GPU_UUID': gpu, 'active_workers': len(active)}), flush=True)
            atomic_json(out / 'inference_supervisor_progress.json', {'active_workers': len(active), 'pending_groups': len(pending), 'completed_groups': len(done),
                'remaining_RAM_reservation_bytes': headroom([v['witness'] for v in active.values()], cfg['host_planned_peak_per_worker_bytes']),
                'seconds': time.monotonic() - started, 'attempt': attempt, 'recovered_without_live_worker_restart': True})
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > cfg['new_stage_disk_ceiling_bytes']:
                raise RuntimeError('Private stage storage guard')
            time.sleep(5)
        records = [frozen.completed_case(out, job, identity) for job in plan['jobs'].values()]
        if any(r is None for r in records):
            raise ValueError('Every actually sealed execution required')
        atomic_json(out / 'full_inference_manifest.json', {'status': 'COMPLETE_FROZEN_FULL_LIMITED_MASA_SEMANTIC_PREDICTIONS_NOT_YET_METRICS',
            'preregistration_commit': commit, 'identity': identity, 'plan_sha256': plan_sha,
            'unique_executions': len(records), 'logical_cases': len(plan['logical_cases']), 'records': records,
            'all988videos': True, 'all304561physicalIDs_each_execution': True, 'Val_GT_or_Val_metrics_in_inference': False,
            'legal_Train_Known_prototypes_only': True, 'TAO_Test_access': False, 'foreign_interference': False,
            'wall_seconds': time.monotonic() - started, 'scheduling_recovery_receipt': f'recovery_{attempt}_started.json',
            'adopted_original_workers_terminal_proof': adopted_completed})
        print(json.dumps({'status': 'COMPLETE_ACTUAL_ALL_REGISTERED_FROZEN_INFERENCE', 'executions': len(records)}), flush=True)
    except Exception as exc:
        error = repr(exc)
        # Only children created by THIS parent are eligible for safety cleanup.
        # Previously orphaned workers and every foreign process remain untouched.
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
        raise
    finally:
        for log in logs:
            log.close()
        legacy_live = [v['witness'] for v in active.values() if v['child'] is None and same_process(v['witness'])]
        atomic_json(out / 'inference_supervisor_progress.json', {'active_workers': len(legacy_live), 'pending_groups': len(pending), 'completed_groups': len(done),
            'remaining_RAM_reservation_bytes': headroom(legacy_live, cfg['host_planned_peak_per_worker_bytes']), 'seconds': time.monotonic() - started, 'attempt': attempt, 'error': error})
        atomic_json(out / f'supervisor_{attempt}.json', {'error': error, 'completed_groups': done,
            'owned_child_returncodes': [p.returncode for p in children], 'adopted_original_workers_terminal_proof': adopted_completed,
            'legacy_live_workers_left_untouched': legacy_live, 'foreign_process_interference': False, 'wall_seconds': time.monotonic() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    main(parser.parse_args().preregistration_commit)
