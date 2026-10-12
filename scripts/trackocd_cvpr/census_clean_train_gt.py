#!/usr/bin/env python3
"""One CPU, remotely preregistered, explicitly user-authorized Known census."""
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
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.scoped_train_census import inherited_known_ids, validate_video_scope, census
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory

SOURCES = ('configs/trackocd_cvpr/clean_train_gt_census_v1.json',
           'configs/trackocd_cvpr/clean_train_gt_scope_approval_v1.json',
           'configs/trackocd_cvpr/clean_train_gt_scope_proposal.json',
           'scripts/trackocd_cvpr/census_clean_train_gt.py',
           'src/trackocd_cvpr/scoped_train_census.py', 'src/trackocd_cvpr/metadata_only.py',
           'scripts/trackocd_cvpr/audit_train_metadata_containment.py',
           'src/trackocd_v2/io.py', 'configs/trackocd_cvpr/autonomous_research.json')


def reject_sensitive_open(event, args, permitted):
    if event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path in permitted:
        return
    if ('annotations' in path.parts or 'frames' in path.parts or 'checkpoints' in path.parts
            or path.name in ('selection_plan.json', 'roles.json', 'train_labels.parquet', 'evaluator_only_geometry_join.json')
            or path.suffix.lower() in ('.parquet', '.npy', '.npz', '.pth', '.pt', '.jpg', '.jpeg', '.png', '.mp4')):
        raise PermissionError('Scoped census rejects other labels/pixels/features/models')


def validate_authorization(cfg, approval, proposal_hash):
    if (approval['status'] != 'USER_APPROVED_SCOPED_EXTERNAL_CENSUS_ONLY'
            or approval['user_message_verbatim'] != '同意上述范围，恢复原 Goal'
            or approval['bound_proposal_sha256'] != proposal_hash or cfg['proposal_sha256'] != proposal_hash
            or approval['external_known_only_Train_census_authorized'] is not True
            or approval['automatic_training_or_full_cache_authorized_by_this_receipt'] is not False
            or (approval['allowlisted_videos'], approval['excluded_old_heldout_videos']) != (429, 51)
            or (cfg['allowlisted_videos'], cfg['excluded_old_heldout_videos']) != (429, 51)
            or approval['all_other_video_IDs_default_deny'] is not True
            or approval['category_names_Novel_vocabulary_video_GT_arrays_Val_Test_other_protected_labels_authorized'] is not False):
        raise ValueError('Explicit fixed-boundary user approval required, not self-authorization')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    parser.add_argument('--git-proxy', default='')
    args = parser.parse_args()
    cfg = json.loads((ROOT / SOURCES[0]).read_text())
    approval = json.loads((ROOT / cfg['approval']).read_text())
    validate_authorization(cfg, approval, sha256_file(ROOT / cfg['proposal']))
    safety = json.loads((ROOT / SOURCES[-1]).read_text())
    git = ['git'] + (['-c', f'http.proxy={args.git_proxy}'] if args.git_proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote = subprocess.check_output(git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"],
                                    cwd=ROOT, text=True, timeout=35,
                                    env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()
    if not remote or remote[0] != args.preregistration_commit:
        raise ValueError('Exact current remote registration required before GT read')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{args.preregistration_commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered source differs')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve all prior attempts')
    annotation = Path(cfg['train_annotation'])
    roles = ROOT / cfg['roles']
    index_path = ROOT / cfg['private_exclusion_index']
    for path in (annotation, roles, index_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Fixed regular census input required')
    if annotation.name != 'train.json':
        raise ValueError('Train only')
    sys.addaudithook(lambda event, values: reject_sensitive_open(event, values, {annotation.resolve(), roles.resolve(), index_path.resolve()}))
    started, peak_rss = time.monotonic(), 0
    project_bytes_before = sum(path.stat().st_size for path in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if path.is_file())

    def guard():
        nonlocal peak_rss
        measured = process_memory()
        peak_rss = max(peak_rss, measured['sampled_smaps_rollup_Rss_bytes'])
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
        if (mem['MemAvailable'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']
                or measured['guarded_process_RSS_bytes'] > cfg['rss_bytes_ceiling']
                or time.monotonic() - started > cfg['wall_seconds_ceiling']
                or shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes']):
            raise RuntimeError('RAM/RSS/wall/free-disk census guard')
        output_bytes = sum(p.stat().st_size for p in out.iterdir() if p.is_file()) if out.exists() else 0
        if (output_bytes > cfg['output_bytes_ceiling']
                or project_bytes_before + output_bytes > safety['incremental_storage_hard_bytes']):
            raise RuntimeError('Small private/public output bound')

    guard()
    out.mkdir(parents=True)
    launch = {'preregistration_commit': args.preregistration_commit,
              'source_sha256': {name: sha256_file(ROOT / name) for name in SOURCES},
              'roles_source_sha256': cfg['roles_sha256'], 'Train_annotation_sha256': cfg['train_annotation_sha256'],
              'private_exclusion_index_sha256': cfg['private_exclusion_index_sha256']}
    atomic_json(out / 'launch.json', launch)
    try:
        for path, digest in ((annotation, cfg['train_annotation_sha256']), (roles, cfg['roles_sha256']),
                             (index_path, cfg['private_exclusion_index_sha256'])):
            if sha256_file(path) != digest:
                raise ValueError('Fixed census input hash differs')
        index = json.loads(index_path.read_text())
        if index['source_plan_sha256'] != cfg['private_index_source_plan_sha256']:
            raise ValueError('Exclusion index plan binding differs')
        allowed, excluded = validate_video_scope(index)
        known = inherited_known_ids(roles)
        guard()
        public, private = census(annotation, allowed, excluded, known, guard)
        # Check inputs remained unchanged, including skipped bytes.
        for path, digest in ((annotation, cfg['train_annotation_sha256']), (roles, cfg['roles_sha256']),
                             (index_path, cfg['private_exclusion_index_sha256'])):
            if sha256_file(path) != digest:
                raise ValueError('Census input changed during read')
        private_path = out / 'private_known_support_index.json'
        atomic_json(private_path, private)
        guard()
        snapshot = process_memory()
        peak_rss = max(peak_rss, snapshot['sampled_smaps_rollup_Rss_bytes'])
        result = {'status': 'COMPLETE_SCOPED_KNOWN_POSITIVE_CENSUS_NOT_PROTOCOL_OR_SCIENTIFIC_ACCEPTANCE',
                  **launch, 'census': public, 'interpretation': cfg['interpretation'],
                  'private_index': {'bytes': private_path.stat().st_size, 'sha256': sha256_file(private_path)},
                  'category_names_Novel_vocabulary_nested_video_GT_protected_target_values_decoded': False,
                  'Val_Test_pixels_features_models_read': False, 'labels_in_forward_or_live_memory': False,
                  'direct_sensitive_open_barrier_not_complete_sandbox': True,
                  'historical_access_incident_erased': False, 'optimization_steps': 0,
                  'resources': {'wall_seconds': time.monotonic() - started, **snapshot,
                                'peak_sampled_smaps_rollup_Rss_bytes_through_result_snapshot': peak_rss,
                                'resource_ru_maxrss_bytes_including_preexec_history': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                'training_GPU_hours': 0, 'disk_free_bytes': shutil.disk_usage(ROOT).free}}
        guard()
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'census': public, 'resources': result['resources']}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error),
                                         'complete': False, 'wall_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
