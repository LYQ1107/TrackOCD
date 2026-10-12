#!/usr/bin/env python3
"""Preregistered administrative projection, no semantic/heldout target labels."""
import argparse
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
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.metadata_only import train_plan_partition_metadata
from src.trackocd_cvpr.train_partition_scope import partition_video_scope
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory

SOURCES = ('configs/trackocd_cvpr/train_partition_scope_audit.json',
           'scripts/trackocd_cvpr/audit_train_partition_scope.py',
           'src/trackocd_cvpr/metadata_only.py', 'src/trackocd_cvpr/train_partition_scope.py',
           'scripts/trackocd_cvpr/audit_train_metadata_containment.py',
           'src/trackocd_cvpr/train_physical_resume.py', 'src/trackocd_core/physical_qualification.py',
           'src/trackocd_v2/io.py', 'configs/trackocd_cvpr/autonomous_research.json')


def reject_sensitive_open(event, args, permitted_plan):
    if event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path == permitted_plan:
        return  # Only the exact hash-bound administrative source is permitted.
    if ('annotations' in path.parts or 'frames' in path.parts or 'checkpoints' in path.parts
            or path.name in ('selection_plan.json', 'roles.json', 'train_labels.parquet', 'evaluator_only_geometry_join.json')
            or path.suffix.lower() in ('.parquet', '.npy', '.npz', '.pth', '.jpg', '.jpeg', '.png', '.mp4')):
        raise PermissionError('Administrative audit rejects semantic labels, pixels, features and models')


def validate_summary(public, cfg):
    if public['selected_rows'] != cfg['expected_selected_rows'] or {
            part: row['selected_rows'] for part, row in public['partitions'].items()} != cfg['expected_partition_rows']:
        raise ValueError('All declared old partition rows must match; no exclusions/repair')


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
        raise ValueError('Exact remote administrative-audit registration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{args.preregistration_commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered source changed')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve all previous attempts')
    plan_path, manifest_path = ROOT / cfg['private_plan'], ROOT / cfg['cache_manifest']
    if plan_path.is_symlink() or manifest_path.is_symlink():
        raise ValueError('Administrative source must be a fixed regular file')
    permitted = plan_path.resolve()
    sys.addaudithook(lambda event, values: reject_sensitive_open(event, values, permitted))
    started, peak_sampled_rss = time.monotonic(), 0
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
            raise RuntimeError('RAM/RSS/wall/free-disk guard')
        if out.exists() and sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > cfg['output_bytes_ceiling']:
            raise RuntimeError('Small administrative output bound')
    guard()
    if (plan_path.stat().st_size != cfg['private_plan_bytes'] or sha256_file(plan_path) != cfg['private_plan_sha256']
            or sha256_file(manifest_path) != cfg['cache_manifest_sha256']):
        raise ValueError('Old sealed plan/manifest identity differs')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['selection_plan_sha256'] != cfg['private_plan_sha256']
            or manifest['preregistration_commit'] != cfg['historical_registration_commit']):
        raise ValueError('Reference cache does not bind the administrative plan')
    out.mkdir(parents=True)
    launch = {'preregistration_commit': args.preregistration_commit,
              'source_sha256': {name: sha256_file(ROOT / name) for name in SOURCES},
              'cache_manifest_sha256': cfg['cache_manifest_sha256'], 'private_plan_sha256': cfg['private_plan_sha256']}
    atomic_json(out / 'launch.json', launch)
    try:
        rows = train_plan_partition_metadata(plan_path)
        guard()
        public, private = partition_video_scope(rows, cfg['expected_selected_rows'])
        validate_summary(public, cfg)
        path = out / 'private_partition_video_index.json'
        atomic_json(path, {'schema_version': 'trackocd.cvpr.private-administrative-video-exclusion.v1',
                          'source_plan_sha256': cfg['private_plan_sha256'], 'partitions': private,
                          'not_model_facing': True, 'semantic_target_labels': False})
        guard()
        snapshot = process_memory()
        peak_sampled_rss = max(peak_sampled_rss, snapshot['sampled_smaps_rollup_Rss_bytes'])
        result = {'status': 'COMPLETE_ADMIN_VIDEO_SCOPE_NOT_GT_CENSUS_OR_SCIENTIFIC_CLEARANCE', **launch,
                  'partition_scope': public,
                  'private_exclusion_index': {'bytes': path.stat().st_size, 'sha256': sha256_file(path)},
                  'row_value_fields_JSON_decoded': cfg['decoded_row_fields'],
                  'category_role_observation_summary_values_JSON_decoded': False,
                  'annotation_target_label_tables_roles_images_features_models_Val_Test_read': False,
                  'direct_sensitive_open_barrier_not_complete_sandbox': True,
                  'historical_commands_executed': False, 'optimization_steps': 0,
                  'scientific_access_clearance_granted': False, 'interpretation': cfg['interpretation'],
                  'resources': {'wall_seconds': time.monotonic() - started, **snapshot,
                                'peak_sampled_smaps_rollup_Rss_bytes_through_result_snapshot': peak_sampled_rss,
                                'resource_ru_maxrss_bytes_including_preexec_history': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                'training_GPU_hours': 0, 'disk_free_bytes': shutil.disk_usage(ROOT).free}}
        guard()
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'partition_scope': public, 'resources': result['resources']}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False})
        raise


if __name__ == '__main__':
    main()
