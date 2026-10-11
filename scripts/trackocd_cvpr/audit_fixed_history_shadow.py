#!/usr/bin/env python3
"""Registered same-history future-input shadow; never alter actual decisions."""
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
from scripts.trackocd_cvpr.audit_historical_full import csv_write
SOURCES = ('configs/trackocd_cvpr/cvpr0_fixed_history_shadow.json',
    'scripts/trackocd_cvpr/audit_fixed_history_shadow.py', 'src/trackocd_cvpr/fixed_history_shadow.py',
    'src/trackocd_cvpr/checkpoint_diagnostics.py', 'src/trackocd_core/persistent_policy.py',
    'src/trackocd_core/train_first_replay.py', 'src/trackocd_core/train_first_experiment.py',
    'configs/trackocd_cvpr/autonomous_research.json')
KEYS = ('representation', 'policy', 'seed', 'partition', 'order', 'prefix')


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
        raise ValueError('Exact remote shadow registration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{remote}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered shadow source differs')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve every actual attempt')
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.trackocd_core.train_first_cache import TrainFirstCache
    from src.trackocd_core.train_first_experiment import frozen_model, evidence_bank, prototypes, evaluate
    from src.trackocd_core.persistent_policy import DecisionMLP
    from src.trackocd_cvpr.fixed_history_shadow import fixed_history_replay
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
        if out.exists() and sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) > cfg['output_bytes_ceiling']:
            raise RuntimeError('Private trace/output bound')
        if sum(f.stat().st_size for f in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if f.is_file()) > safety['incremental_storage_hard_bytes']:
            raise RuntimeError('CVPR retained incremental storage ceiling')

    guard()
    historical = ROOT / 'outputs/trackocd_core/core_training/train_first_v1'
    freeze_path = historical / 'model_freeze.json'
    if sha256_file(freeze_path) != cfg['model_freeze_sha256']:
        raise ValueError('Protected model registry differs')
    for name, digest in json.loads(freeze_path.read_text())['protected_sha256'].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError('Protected source/input differs')
    cache_root = ROOT / 'outputs/trackocd_core/features/train_first_v1'
    if sha256_file(cache_root / 'manifest.json') != cfg['feature_manifest_sha256']:
        raise ValueError('Frozen Train feature universe differs')
    cache = TrainFirstCache(cache_root)
    source = pq.read_table(cache_root / 'train_labels.parquet',
                           filters=[('partition', 'in', cfg['allowed_label_partitions'])]).to_pylist()
    route_by = {r['key']: r for r in cache.rows}
    labels = [{**r, 'video_id': route_by[r['key']]['video_id']} for r in source]
    by = {r['key']: r for r in labels}
    known = sorted({r['category_id'] for r in labels if r['partition'] == 'representation_fit'})
    parts = {part: [r for r in cache.rows if r['key'] in by and by[r['key']]['partition'] == part] for part in cfg['partitions']}
    if {k: len(v) for k, v in parts.items()} != {'development': 258, 'policy_train': 229}:
        raise ValueError('Frozen diagnostic partitions differ')
    receipt = json.loads((historical / 'policy/training_receipt.json').read_text())
    fits = {(f['representation'], f['seed'], f['model']): f for f in receipt['fits']}
    constants = json.loads((ROOT / 'configs/trackocd_core/persistent_policy_training.json').read_text())['risk_constants']
    reference_path = ROOT / 'outputs/trackocd_cvpr/cvpr0/state_trajectories_v1/result.json'
    if sha256_file(reference_path) != cfg['state_result_sha256']:
        raise ValueError('Actual unchanged state reference differs')
    reference_result = json.loads(reference_path.read_text())
    reference_cases = {tuple(c[k] for k in KEYS): c for c in reference_result['cases']}
    private_path = reference_path.parent / reference_result['private_trace']['filename']
    if sha256_file(private_path) != reference_result['private_trace']['sha256']:
        raise ValueError('Retained actual state trace differs')
    original_case_path = ROOT / 'outputs/trackocd_cvpr/cvpr0/checkpoint_replay_v1/result.json'
    if sha256_file(original_case_path) != reference_result['reference_result_sha256']:
        raise ValueError('Reference evaluator result differs')
    original = {tuple(c[k] for k in KEYS): c for c in json.loads(original_case_path.read_text())['cases']
                if c['intervention'] == 'ACTUAL_FROZEN_CHECKPOINT' and 'diagnostic_condition' not in c}
    out.mkdir(parents=True)
    launch = {'preregistration_commit': remote, 'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES},
        'actual_state_reference_sha256': cfg['state_result_sha256'], 'config_sha256': sha256_file(ROOT / SOURCES[0])}
    atomic_json(out / 'launch.json', launch)
    cases, total, csv_rows = [], 0, []
    trace_path = out / 'private_fixed_history_shadow.parquet'
    writer = None
    try:
        for representation in cfg['representations']:
            for seed in cfg['seeds']:
                guard()
                rep, lineage = frozen_model(ROOT, representation, seed)
                proto = prototypes(cache, labels, known, rep)
                fit = fits[(representation, seed, cfg['policy'])]
                cp = fit['selected_checkpoint']['checkpoint']
                if sha256_file(ROOT / cp['path']) != cp['sha256']:
                    raise ValueError('Frozen policy checkpoint differs')
                model = DecisionMLP().eval().requires_grad_(False)
                model.load_state_dict(torch.load(ROOT / cp['path'], map_location='cpu', weights_only=True)['state_dict'])
                bias = fit['operating_points']['1.0']['wait_bias']
                for part, routes in parts.items():
                    bank = evidence_bank(cache, routes, rep)
                    order = sorted({r['video_id'] for r in routes})
                    for prefix in cfg['prefixes']:
                        guard()
                        identity = dict(zip(KEYS, (representation, cfg['policy'], seed, part, 'main', prefix)))
                        sealed, trace, summary = fixed_history_replay(routes, order, prefix, proto,
                            bank[prefix], bank[cfg['future_input_cap_evaluator_only']], model, by, constants, bias)
                        key = tuple(identity[k] for k in KEYS)
                        if not reference_cases[key]['exact_retained_logits_actions_order_and_evaluator_parity']:
                            raise ValueError('Original reference was not a parity-verified actual replay')
                        if evaluate(sealed, routes, labels) != original[key]['evaluation']:
                            raise AssertionError('Future shadow altered actual evaluator outputs')
                        previous = pq.read_table(private_path, use_threads=False,
                            filters=[(k, '=', identity[k]) for k in KEYS]).to_pylist()
                        if len(previous) != len(trace):
                            raise AssertionError('Actual trace universe differs')
                        for row, old, event in zip(trace, previous, sealed.events):
                            if ((row['sequence'], row['physical_key'], row['actual_action'], row['actual_logits']) !=
                                (old['sequence'], old['physical_key'], old['action'], old['logits'])
                                or event.token != old['selected_token']
                                or event.known_category_id != old['selected_known_id']):
                                raise AssertionError('Actual actions/logits/order/tokens changed under shadow')
                        table = pa.Table.from_pylist([{**identity, **r} for r in trace])
                        if writer is None:
                            writer = pq.ParquetWriter(trace_path, table.schema, compression='zstd')
                        writer.write_table(table)
                        total += len(trace)
                        cases.append({**identity, 'wait_bias': bias, 'representation_lineage': lineage,
                            'summary': summary, 'actual_retained_actions_logits_tokens_order_and_metrics_equal': True,
                            'future_shadow_wrote_memory_or_observer': False})
                        csv_rows.append({**identity, **summary})
                        atomic_json(out / 'progress.json', {'status': 'RUNNING_TRAIN_FIXED_HISTORY_SHADOW', 'cases': len(cases), 'rows': total})
    finally:
        if writer is not None:
            writer.close()
    guard()
    if (len(cases), total) != (cfg['expected_cases'], cfg['expected_rows']):
        raise ValueError('Incomplete registered future-shadow universe')
    # Sparse Counters differ by case; normalize columns without inventing observations.
    fields = set().union(*(r.keys() for r in csv_rows))
    csv_write(out / 'fixed_history_shadow_summary.csv', [{k: r.get(k, 0) for k in sorted(fields)} for r in csv_rows])
    result = {'status': 'COMPLETE_TRAIN_FIXED_HISTORY_FUTURE_SHADOW_NOT_WAITING_POLICY_PROOF', **launch,
        'cases': cases, 'rows': total,
        'private_trace': {'filename': trace_path.name, 'bytes': trace_path.stat().st_size, 'sha256': sha256_file(trace_path)},
        'GT_in_live_memory_or_forward': False, 'future_features_in_actual_forward_or_memory': False,
        'unique_early_commit_causality_or_deployable_wait_gain_proven': False,
        'Val_or_Test_access': False, 'final_holdout_labels_read': False, 'optimization_steps': 0,
        'resources': {'wall_seconds': time.monotonic() - started,
            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'training_GPU_hours': 0},
        'interpretation': cfg['interpretation']}
    atomic_json(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 'cases': len(cases), 'rows': total, 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    main()
