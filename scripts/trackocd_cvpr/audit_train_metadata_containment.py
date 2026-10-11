#!/usr/bin/env python3
"""Bounded containment: sealed sanitized plan and source bytes, not GT data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_cvpr.train_physical_resume import validate_plan
from src.trackocd_v2.io import atomic_json, sha256_file

SOURCES = ('configs/trackocd_cvpr/metadata_containment_audit.json',
           'scripts/trackocd_cvpr/audit_train_metadata_containment.py',
           'src/trackocd_cvpr/train_physical_resume.py', 'src/trackocd_core/physical_qualification.py',
           'src/trackocd_cvpr/metadata_only.py', 'src/trackocd_v2/io.py',
           'outputs/trackocd_cvpr/safety/train_nested_metadata_access_incident_v1.json',
           'configs/trackocd_cvpr/autonomous_research.json')


def reject_sensitive_open(event, args):
    """Additional direct-open barrier, not a complete subprocess sandbox."""
    if event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0]))
    if ('annotations' in path.parts or 'frames' in path.parts or 'checkpoints' in path.parts
            or path.name in ('train_labels.parquet', 'roles.json', 'selection_plan.json', 'evaluator_only_geometry_join.json')
            or path.suffix.lower() in ('.npz', '.npy', '.pth', '.jpg', '.jpeg', '.png', '.mp4')):
        raise PermissionError('Containment audit rejects annotation/label/pixel/feature/model reads')


def process_memory(status_text=None, rollup_text=None):
    status = Path('/proc/self/status').read_text() if status_text is None else status_text
    rollup = Path('/proc/self/smaps_rollup').read_text() if rollup_text is None else rollup_text
    def field(text, key):
        lines = [line.split() for line in text.splitlines() if line.startswith(key + ':')]
        if len(lines) != 1 or len(lines[0]) != 3 or lines[0][2] != 'kB' or int(lines[0][1]) <= 0:
            raise RuntimeError('Missing/invalid own process memory counter')
        return int(lines[0][1]) * 1024
    hwm, rss = field(status, 'VmHWM'), field(rollup, 'Rss')
    return {'kernel_reported_VmHWM_bytes_approximate': hwm, 'sampled_smaps_rollup_Rss_bytes': rss,
            'guarded_process_RSS_bytes': max(hwm, rss)}


def inspect_sanitized_plan(plan, cfg):
    if set(plan) != {'schema_version', 'videos', 'GT_or_category_fields', 'cadence', 'selection'}:
        raise ValueError('Unexpected sealed-plan top-level fields')
    if plan['schema_version'] != 'trackocd.cvpr0.private-train-image-plan.v1':
        raise ValueError('Unexpected sealed-plan schema')
    videos = validate_plan(plan, {'videos': cfg['expected_videos'], 'canonical_images': cfg['expected_images']})
    return {'videos': len(videos), 'images': sum(len(v['images']) for v in videos),
            'exact_sanitized_plan_schema': True, 'nested_GT_fields_in_sanitized_plan': False,
            'no_category_or_video_values_published': True}


def historical_sources(root, proof, cfg):
    if proof['preregistration_commit'] != cfg['physical_registration_commit']:
        raise ValueError('Existing physical source registration differs')
    records = []
    for name, digest in proof['source_sha256'].items():
        source = subprocess.check_output(['git', 'show', f"{proof['preregistration_commit']}:{name}"], cwd=root)
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError('Historical recorded source-file binding differs')
        records.append({'path': name, 'recorded_source_file_binding_matches_git': True,
                        'historical_loaded_runtime_module_identity_proven': False})
    source = subprocess.check_output(['git', 'show', f"{proof['preregistration_commit']}:{cfg['physical_config']}"], cwd=root)
    physical = json.loads(source)
    if physical['private_plan_sha256'] != cfg['private_sanitized_plan_sha256']:
        raise ValueError('Physical source configuration does not bind sealed sanitized plan')
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    parser.add_argument('--git-proxy', default='')
    args = parser.parse_args()
    cfg = json.loads((ROOT / SOURCES[0]).read_text())
    safety = json.loads((ROOT / SOURCES[-1]).read_text())
    git = ['git'] + (['-c', f'http.proxy={args.git_proxy}'] if args.git_proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote = subprocess.check_output(git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"],
        cwd=ROOT, text=True, timeout=35, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()
    if not remote or remote[0] != args.preregistration_commit:
        raise ValueError('Exact current remote safety preregistration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{args.preregistration_commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered safety source changed')
    sys.addaudithook(reject_sensitive_open)
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Never overwrite containment attempts')
    started = time.monotonic()
    peak_sampled_rss = 0
    def guard():
        nonlocal peak_sampled_rss
        measured = process_memory()
        peak_sampled_rss = max(peak_sampled_rss, measured['sampled_smaps_rollup_Rss_bytes'])
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
        if (mem['MemAvailable'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']
                or measured['guarded_process_RSS_bytes'] > cfg['rss_bytes_ceiling']
                or time.monotonic() - started > cfg['wall_seconds_ceiling']
                or shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes']):
            raise RuntimeError('Safety RAM/RSS/wall/disk bound')
        if out.exists() and sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > cfg['output_bytes_ceiling']:
            raise RuntimeError('Small containment output bound')
    guard()
    out.mkdir(parents=True)
    launch = {'preregistration_commit': args.preregistration_commit,
              'source_sha256': {name: sha256_file(ROOT / name) for name in SOURCES},
              'incident_sha256': sha256_file(ROOT / cfg['incident_path'])}
    atomic_json(out / 'launch.json', launch)
    try:
        result_path, plan_path = ROOT / cfg['preflight_result'], ROOT / cfg['private_sanitized_plan']
        if (sha256_file(result_path) != cfg['preflight_result_sha256']
                or plan_path.stat().st_size != cfg['private_sanitized_plan_bytes']
                or sha256_file(plan_path) != cfg['private_sanitized_plan_sha256']):
            raise ValueError('Existing sealed result/plan changed; no replacement permitted')
        preflight = json.loads(result_path.read_text())
        if preflight['private_plan']['sha256'] != cfg['private_sanitized_plan_sha256']:
            raise ValueError('Preflight does not bind sanitized plan')
        plan = inspect_sanitized_plan(json.loads(plan_path.read_text()), cfg)
        registration_path = ROOT / cfg['physical_registration']
        if sha256_file(registration_path) != cfg['physical_registration_sha256']:
            raise ValueError('Sealed physical registration changed')
        proof = json.loads(registration_path.read_text())
        sources = historical_sources(ROOT, proof, cfg)
        guard()
        result = {'status': 'PASS_SEALED_PLAN_SOURCE_CONTAINMENT_NOT_SCIENTIFIC_CLEARANCE', **launch,
                  'sanitized_plan_sha256': cfg['private_sanitized_plan_sha256'], 'sanitized_plan': plan,
                  'recorded_physical_sources': sources, 'prior_GT_metadata_access_undone': False,
                  'all_historical_runtime_leakage_absent_proven': False, 'scientific_access_clearance_granted': False,
                  'direct_sensitive_open_barrier_installed_not_complete_sandbox': True,
                  'annotation_target_labels_images_features_models_Val_Test_read': False,
                  'optimization_steps': 0, 'interpretation': cfg['interpretation'],
                  'resources': {'wall_seconds': time.monotonic() - started,
                                **process_memory(), 'peak_sampled_smaps_rollup_Rss_bytes': peak_sampled_rss,
                                'resource_ru_maxrss_bytes_including_preexec_history': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                'rss_measurement': cfg['rss_measurement'],
                                'training_GPU_hours': 0, 'disk_free_bytes': shutil.disk_usage(ROOT).free}}
        atomic_json(out / 'result.json', result)
        guard()
        print(json.dumps({'status': result['status'], 'sanitized_plan': plan, 'resources': result['resources']}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False})
        raise


if __name__ == '__main__':
    main()
