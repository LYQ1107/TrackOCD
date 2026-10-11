#!/usr/bin/env python3
"""Frozen all-ID semantic/seal/evaluator engineering on capped Known reference."""
from collections import Counter, defaultdict
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
SOURCES = ('configs/trackocd_cvpr/cvpr0_tiny_semantic_integration.json',
    'scripts/trackocd_cvpr/integrate_tiny_train_semantics.py',
    'src/trackocd_cvpr/truncated_reference_join.py', 'src/trackocd_core/train_first_replay.py',
    'src/trackocd_core/train_first_experiment.py', 'src/trackocd_core/limited_masa_features.py',
    'src/trackocd_core/sealed_parquet.py', 'src/trackocd_core/evaluation/contracts.py',
    'src/trackocd_core/evaluation/standard.py', 'src/trackocd_core/evaluation/persistent.py')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preregistration-commit', required=True)
    p.add_argument('--git-proxy', default='')
    args = p.parse_args()
    cfg = json.loads((ROOT / SOURCES[0]).read_text())
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    git = ['git'] + (['-c', f'http.proxy={args.git_proxy}'] if args.git_proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote = subprocess.check_output(git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"], text=True,
        timeout=35, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()[0]
    if remote != args.preregistration_commit:
        raise ValueError('Exact remote tiny integration registration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{remote}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered integration source differs')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve all actual attempts')
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    from src.trackocd_core.train_first_cache import TrainFirstCache
    from src.trackocd_core.train_first_experiment import frozen_model, evidence_bank, prototypes
    from src.trackocd_core.train_first_replay import replay
    from src.trackocd_core.persistent_policy import DecisionMLP
    from src.trackocd_core.limited_masa_features import LimitedMasaVideoCache
    from src.trackocd_core.sealed_parquet import write_sealed, read_sealed
    from src.trackocd_core.evaluation import Target, TrackKey, join_evaluation, evaluate_standard, evaluate_persistent
    from src.trackocd_cvpr.truncated_reference_join import truncated_join
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()

    def guard():
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']:
            raise RuntimeError('RAM reserve')
        if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg['rss_bytes_ceiling']
                or time.monotonic() - started > cfg['wall_seconds_ceiling']
                or shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes']):
            raise RuntimeError('RSS/wall/disk ceiling')
        if out.exists() and sum(f.stat().st_size for f in (out / 'private_ledgers').rglob('*') if f.is_file()) > cfg['private_ledger_bytes_ceiling']:
            raise RuntimeError('Private ledger output bound')

    guard()
    historical = ROOT / 'outputs/trackocd_core/core_training/train_first_v1'
    freeze_path = historical / 'model_freeze.json'
    if sha256_file(freeze_path) != cfg['model_freeze_sha256']:
        raise ValueError('Protected model registry differs')
    for name, digest in json.loads(freeze_path.read_text())['protected_sha256'].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError('Protected input/source/model differs')
    cache_root = ROOT / 'outputs/trackocd_core/features/train_first_v1'
    if sha256_file(cache_root / 'manifest.json') != cfg['feature_manifest_sha256']:
        raise ValueError('Cached GT reference differs')
    cache = TrainFirstCache(cache_root)
    source = pq.read_table(cache_root / 'train_labels.parquet',
                           filters=[('partition', 'in', cfg['allowed_label_partitions'])]).to_pylist()
    route_by = {r['key']: r for r in cache.rows}
    labels = [{**r, 'video_id': route_by[r['key']]['video_id']} for r in source]
    by = {r['key']: r for r in labels}
    known = sorted({r['category_id'] for r in labels if r['partition'] == 'representation_fit'})
    gt_rows = [r for r in cache.rows if r['key'] in by and by[r['key']]['partition'] == 'development' and r['video_id'] == cfg['video_id']]
    if len(gt_rows) != cfg['expected_GT_reference_tracks'] or any(by[r['key']]['simulation_role'] != 'known' for r in gt_rows):
        raise ValueError('Registered selected/capped Known-only reference differs')
    recovered_path = ROOT / cfg['recovered_directory'] / 'result.json'
    if sha256_file(recovered_path) != cfg['recovered_result_sha256']:
        raise ValueError('Recovered valid input receipt differs')
    recovered = json.loads(recovered_path.read_text())
    if recovered['status'] != 'RECOVERED_VALID_ALL_ID_FEATURE_INPUT_AFTER_FAILED_FINAL_JSON_NOT_NORMAL_WORKER_SUCCESS':
        raise ValueError('Only the verified recovery input')
    payload = recovered['private_feature_payload']
    path = ROOT / payload['directory'] / payload['filename']
    if sha256_file(path) != cfg['feature_sha256'] or payload['sha256'] != cfg['feature_sha256']:
        raise ValueError('Predicted all-ID input differs')
    pred_cache = LimitedMasaVideoCache(path, cfg['video_id'])
    pred_rows = pred_cache.rows
    if len(pred_rows) != cfg['expected_predicted_tracks']:
        raise ValueError('All predicted IDs required')
    receipt = json.loads((historical / 'policy/training_receipt.json').read_text())
    fits = {(f['representation'], f['seed'], f['model']): f for f in receipt['fits']}
    out.mkdir(parents=True)
    (out / 'private_ledgers').mkdir()
    sealed_cases = []
    # All actual predictions/GT-controlled counterfactuals seal BEFORE geometry matching/evaluation.
    for representation in cfg['representations']:
        for seed in cfg['seeds']:
            guard()
            rep, lineage = frozen_model(ROOT, representation, seed)
            proto = prototypes(cache, labels, known, rep)
            cp = fits[(representation, seed, cfg['policy'])]['selected_checkpoint']['checkpoint']
            if sha256_file(ROOT / cp['path']) != cp['sha256']:
                raise ValueError('Frozen checkpoint differs')
            model = DecisionMLP().eval().requires_grad_(False)
            model.load_state_dict(torch.load(ROOT / cp['path'], map_location='cpu', weights_only=True)['state_dict'])
            bias = fits[(representation, seed, cfg['policy'])]['operating_points']['1.0']['wait_bias']
            pred_bank, gt_bank = evidence_bank(pred_cache, pred_rows, rep), evidence_bank(cache, gt_rows, rep)
            for prefix in cfg['prefixes']:
                guard()
                identity = f'{representation}_{seed}_p{prefix}'
                case = {'representation': representation, 'seed': seed, 'prefix': prefix, 'wait_bias': bias, 'lineage': lineage}
                for tag, routes, bank in (('predicted', pred_rows, pred_bank), ('GT_controlled', gt_rows, gt_bank)):
                    sealed, runtime = replay(routes, [cfg['video_id']], prefix, proto, {}, bank[prefix],
                                             decision_model=model, wait_bias=bias)
                    ledger = out / 'private_ledgers' / f'{identity}_{tag}.parquet'
                    write_sealed(ledger, sealed)
                    if read_sealed(ledger, [cfg['video_id']], tuple(proto)).events != sealed.events:
                        raise AssertionError('Real seal/reader roundtrip parity')
                    if len(sealed.events) != len(routes):
                        raise AssertionError('No unmatched/short predicted ID filtering')
                    case[tag] = {'runtime': runtime, 'action_counts': dict(Counter(e.kind for e in sealed.events)),
                        'ledger': {'filename': ledger.name, 'bytes': ledger.stat().st_size, 'sha256': sha256_file(ledger)}}
                sealed_cases.append(case)
    gt_geometry, pred_geometry = defaultdict(list), defaultdict(list)
    gt_keys = {}
    for row in gt_rows:
        gt_keys[row['key']] = TrackKey(row['video_id'], str(row['physical_track_id']))
        view = cache.get_prefix(row['key'], 16)
        for frame, box in zip(row['frame_ids'], view.boxes_normalized_xyxy):
            gt_geometry[int(frame)].append((row['key'], box))
    pred_keys = {}
    for row in pred_rows:
        pred_keys[row['key']] = TrackKey(row['video_id'], str(row['physical_track_id']))
        start, count = row['observation_offset'], row['observation_count']
        for frame, box in zip(pred_cache.data['frame_index'][start:start + count], pred_cache.data['geometry'][start:start + count]):
            pred_geometry[int(frame)].append((row['key'], box))
    match_cfg = cfg['matching']
    matched, match_audit = truncated_join(gt_geometry, pred_geometry, match_cfg['IoU_threshold'],
                                         match_cfg['minimum_shared_observations'], match_cfg['minimum_fraction_of_capped_GT'])
    targets = [Target(gt_keys[r['key']], int(by[r['key']]['category_id']), 'known') for r in gt_rows]
    matches = {pred_keys[k]: gt_keys[matched[k]] if k in matched else None for k in pred_keys}
    for case in sealed_cases:
        for tag in ('predicted', 'GT_controlled'):
            ledger = out / 'private_ledgers' / case[tag]['ledger']['filename']
            sealed = read_sealed(ledger, [cfg['video_id']], known)
            join = join_evaluation(sealed, targets, matches if tag == 'predicted' else {t.key: t.key for t in targets})
            standard, persistent = evaluate_standard(join), evaluate_persistent(join)
            standard.pop('global_anonymous_hungarian_mapping_evaluator_only')
            if (standard['old_denominator'], standard['new_denominator'], standard['h_score'], persistent['commit_ct_denominator']) != (7, 0, None, 0):
                raise AssertionError('Selected reference denominator/Novel unavailable convention')
            case[tag]['evaluation'] = {'standard': standard, 'persistent': persistent}
    guard()
    if len(sealed_cases) != cfg['expected_cases']:
        raise ValueError('Incomplete tiny engineering integration')
    pred_cache.close()
    result = {'status': 'PASS_TINY_ALL_ID_SEMANTIC_TRUNCATED_REFERENCE_INTEGRATION_NOT_MAIN_METRICS',
        'preregistration_commit': remote, 'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES},
        'recovered_feature_result_sha256': cfg['recovered_result_sha256'], 'cases': sealed_cases,
        'all_predicted_IDs_sealed_before_reference_matching': True, 'GT_in_predicted_inference': False,
        'selected_truncated_reference': {'targets': 7, 'Known': 7, 'Novel': 0, 'cross_video_reuse_opportunities': 0},
        'matching': match_audit, 'all_missing_GT_references_in_denominator': True,
        'no_Novel_H_or_CT_gain_claim': True, 'old_numeric_ID_routing_not_repaired': True,
        'GT_counterfactual_changes_input_stream_and_routing_not_unique_causal_proof': True,
        'Val_or_Test_access': False, 'final_holdout_labels_read': False, 'optimization_steps': 0,
        'resources': {'wall_seconds': time.monotonic() - started,
            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'training_GPU_hours': 0},
        'interpretation': cfg['interpretation']}
    atomic_json(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 'cases': len(sealed_cases), 'matching': match_audit, 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    main()
