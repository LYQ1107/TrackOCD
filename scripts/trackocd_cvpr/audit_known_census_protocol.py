#!/usr/bin/env python3
"""Registered external Known-index feasibility, not a training launcher."""
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
from src.trackocd_cvpr.known_census_protocol import known_presence, fixed_hypotheses
from src.trackocd_cvpr.scoped_train_census import inherited_known_ids, validate_video_scope
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory
from src.trackocd_v2.io import atomic_json, sha256_file

SOURCES = ('configs/trackocd_cvpr/known_census_protocol_feasibility_v1.json',
           'scripts/trackocd_cvpr/audit_known_census_protocol.py',
           'src/trackocd_cvpr/known_census_protocol.py', 'src/trackocd_cvpr/disjoint_video_witness.py',
           'src/trackocd_cvpr/scoped_train_census.py', 'src/trackocd_cvpr/metadata_only.py',
           'scripts/trackocd_cvpr/audit_train_metadata_containment.py', 'src/trackocd_v2/io.py',
           'configs/trackocd_cvpr/clean_train_gt_scope_approval_v1.json',
           'outputs/trackocd_cvpr/cvpr2/clean_train_gt_census_v1/result.json',
           'configs/trackocd_cvpr/autonomous_research.json')


def exact_open_barrier(event, args, inputs, proc, out):
    if event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path in inputs | proc or path.parent == out:
        return
    raise PermissionError('External protocol audit permits only exact registered inputs/proc/output; not a full subprocess sandbox')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    parser.add_argument('--git-proxy', default='')
    args = parser.parse_args()
    cfg = json.loads((ROOT / SOURCES[0]).read_text())
    safety = json.loads((ROOT / SOURCES[-1]).read_text())
    approval = json.loads((ROOT / cfg['approval']).read_text())
    if (approval['external_known_only_Train_census_authorized'] is not True
            or approval['allowlisted_videos'] != 429 or approval['excluded_old_heldout_videos'] != 51
            or approval['category_names_Novel_vocabulary_video_GT_arrays_Val_Test_other_protected_labels_authorized'] is not False):
        raise ValueError('Original actual-user Known-only scope required')
    git = ['git'] + (['-c', f'http.proxy={args.git_proxy}'] if args.git_proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote = subprocess.check_output(git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"], cwd=ROOT,
                                    text=True, timeout=35, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()
    if not remote or remote[0] != args.preregistration_commit:
        raise ValueError('Exact current remote registration required before private labels open')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{args.preregistration_commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered source differs')
    out = (ROOT / cfg['output_directory']).resolve()
    if out.exists():
        raise ValueError('Preserve all previous protocol attempts')
    names = ('private_Known_index', 'private_exclusion_index', 'roles', 'census_result')
    paths = {name: ROOT / cfg[name] for name in names}
    for path in paths.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError('Fixed regular protocol input required')
    permitted = {p.resolve() for p in paths.values()} | {(ROOT / name).resolve() for name in SOURCES}
    proc = {Path(name).resolve() for name in ('/proc/meminfo', '/proc/self/status', '/proc/self/smaps_rollup')}
    sys.addaudithook(lambda event, values: exact_open_barrier(event, values, permitted, proc, out))
    started, peak = time.monotonic(), 0
    project_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file())

    def guard():
        nonlocal peak
        measured = process_memory()
        peak = max(peak, measured['sampled_smaps_rollup_Rss_bytes'])
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
        output_bytes = sum(p.stat().st_size for p in out.iterdir() if p.is_file()) if out.exists() else 0
        if (mem['MemAvailable'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']
                or measured['guarded_process_RSS_bytes'] > cfg['rss_bytes_ceiling']
                or time.monotonic() - started > cfg['wall_seconds_ceiling']
                or shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes']
                or output_bytes > cfg['output_bytes_ceiling']
                or project_bytes + output_bytes > safety['incremental_storage_hard_bytes']):
            raise RuntimeError('Protocol RAM/RSS/wall/disk/output guard')

    guard()
    out.mkdir(parents=True)
    launch = {'preregistration_commit': args.preregistration_commit,
              'source_sha256': {name: sha256_file(ROOT / name) for name in SOURCES},
              'private_input_sha256': {name: cfg[name + '_sha256'] for name in names}}
    atomic_json(out / 'launch.json', launch)
    try:
        for name, path in paths.items():
            if sha256_file(path) != cfg[name + '_sha256']:
                raise ValueError('Fixed protocol input hash differs')
        if paths['private_Known_index'].stat().st_size != cfg['private_Known_index_bytes']:
            raise ValueError('Private label-bearing index size differs')
        prior = json.loads(paths['census_result'].read_text())
        if (prior['private_index']['sha256'] != cfg['private_Known_index_sha256']
                or prior['private_exclusion_index_sha256'] != cfg['private_exclusion_index_sha256']
                or prior['roles_source_sha256'] != cfg['roles_sha256']
                or prior['census']['allowlisted_videos'] != 429):
            raise ValueError('Actual census does not bind private support index')
        allowed, excluded = validate_video_scope(json.loads(paths['private_exclusion_index'].read_text()))
        known = inherited_known_ids(paths['roles'])
        presence, counts = known_presence(json.loads(paths['private_Known_index'].read_text()), allowed, known)
        if counts != cfg['expected']:
            raise ValueError('Every census track/observation/video/class must reconcile; no trimming')
        guard()
        summaries, private = [], []
        for threshold in cfg['two_fixed_eligibility_thresholds']:
            public, assignments = fixed_hypotheses(presence, seed=cfg['fixed_seed'], folds=cfg['folds'],
                policy_classes=cfg['policy_pseudo_classes'], minimum_fit=cfg['minimum_fit_videos'],
                minimum_eval=cfg['minimum_eval_videos'], eligibility_video_support=threshold)
            summaries.append(public)
            private.append({'eligibility_video_support': threshold, 'folds': assignments})
            guard()
        for name, path in paths.items():
            if sha256_file(path) != cfg[name + '_sha256']:
                raise ValueError('Protocol input changed during audit')
        private_path = out / 'private_hypotheses_and_video_witnesses.json'
        atomic_json(private_path, {'not_model_facing': True, 'not_a_frozen_protocol': True, 'hypotheses': private})
        guard()
        snapshot = process_memory()
        peak = max(peak, snapshot['sampled_smaps_rollup_Rss_bytes'])
        result = {'status': 'COMPLETE_FIXED_POSITIVE_KNOWN_FEASIBILITY_NOT_FROZEN_PROTOCOL_OR_MODEL_RESULT',
                  **launch, 'census_reconciliation': counts, 'hypotheses': summaries,
                  'interpretation': cfg['interpretation'], 'aggregate_support_previously_observed_not_blind_preregistration': True,
                  'private_output': {'bytes': private_path.stat().st_size, 'sha256': sha256_file(private_path)},
                  'annotation_old_target_tables_names_Novel_Val_Test_pixels_features_models_read': False,
                  'labels_in_forward_or_live_memory': False, 'optimization_steps': 0,
                  'resources': {'wall_seconds': time.monotonic() - started, **snapshot,
                                'peak_sampled_smaps_rollup_Rss_bytes_through_result_snapshot': peak,
                                'resource_ru_maxrss_bytes_including_preexec_history': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                'training_GPU_hours': 0, 'disk_free_bytes': shutil.disk_usage(ROOT).free}}
        guard()
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'cases': [
            {'support_threshold': h['eligibility_video_support'], 'classes': h['class_pool'], 'folds': [
                {'fold': f['fold'], 'necessary_deficits': f['individual_necessary_support_deficits'],
                 'required_slots': f['required_category_video_slots'], 'matched_slots': f['matched_slots'],
                 'sufficient_witness': f['globally_video_disjoint_sufficient_witness_found']} for f in h['folds']]}
            for h in summaries], 'resources': result['resources']}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error),
                                         'complete': False, 'wall_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
