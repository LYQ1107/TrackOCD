#!/usr/bin/env python3
"""Finite, registered Train-only CVPR-0 diagnostic; no optimizer/Val/Test."""
from __future__ import annotations
import argparse
from collections import Counter
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

SOURCES = ('configs/trackocd_cvpr/autonomous_research.json',
           'configs/trackocd_cvpr/cvpr0_checkpoint_replay.json',
           'src/trackocd_cvpr/checkpoint_diagnostics.py',
           'scripts/trackocd_cvpr/replay_frozen_checkpoints.py')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preregistration-commit', required=True)
    p.add_argument('--git-proxy', default='')
    args = p.parse_args()
    cfg = json.loads((ROOT / SOURCES[1]).read_text())
    safety = json.loads((ROOT / SOURCES[0]).read_text())
    git = ['git'] + (['-c', f'http.proxy={args.git_proxy}'] if args.git_proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote = subprocess.check_output(git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"],
                                     timeout=35, text=True, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()[0]
    if remote != args.preregistration_commit:
        raise ValueError('Exact remote checkpoint diagnostic registration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{remote}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Diagnostic source changed')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve actual attempts; no overwrite')
    import numpy as np
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.trackocd_core.train_first_cache import TrainFirstCache
    from src.trackocd_core.train_first_experiment import frozen_model, evidence_bank, prototypes, evaluate
    from src.trackocd_core.persistent_policy import DecisionMLP
    from scripts.trackocd_core.run_gt_pilot_baselines import registered_orders
    from src.trackocd_cvpr.checkpoint_diagnostics import diagnostic_replay, geometry_diagnostics, checkpoint_gradient_replay
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()

    def guard():
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable'] < mem['MemTotal'] * safety['system_ram_reserve_fraction']:
            raise RuntimeError('RAM reserve')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg['rss_bytes_ceiling']:
            raise RuntimeError('Process RSS ceiling')
        if time.monotonic() - started > cfg['wall_seconds_ceiling']:
            raise RuntimeError('Wall ceiling')
        if shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes']:
            raise RuntimeError('Disk reserve')
        if out.exists() and sum(f.stat().st_size for f in out.iterdir() if f.is_file()) > cfg['output_bytes_ceiling']:
            raise RuntimeError('Output payload ceiling')

    guard()
    historical = ROOT / 'outputs/trackocd_core/core_training/train_first_v1'
    freeze_path = historical / 'model_freeze.json'
    if sha256_file(freeze_path) != cfg['model_freeze_sha256']:
        raise ValueError('Frozen historical model registry changed')
    freeze = json.loads(freeze_path.read_text())
    for name, digest in freeze['protected_sha256'].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError('Protected historical input changed')
    cache_root = ROOT / 'outputs/trackocd_core/features/train_first_v1'
    if sha256_file(cache_root / 'manifest.json') != cfg['feature_manifest_sha256']:
        raise ValueError('Frozen feature manifest differs')
    cache = TrainFirstCache(cache_root)
    # Predicate filtering: do NOT open old heldout_selection labels or a new holdout.
    source = pq.read_table(cache_root / 'train_labels.parquet',
                           filters=[('partition', 'in', cfg['allowed_label_partitions'])]).to_pylist()
    routes_by_key = {r['key']: r for r in cache.rows}
    labels = [{**r, 'video_id': routes_by_key[r['key']]['video_id']} for r in source]
    by = {r['key']: r for r in labels}
    known = sorted({r['category_id'] for r in labels if r['partition'] == 'representation_fit'})
    partitions = {part: [r for r in cache.rows if r['key'] in by and by[r['key']]['partition'] == part]
                  for part in cfg['partitions']}
    if {k: len(v) for k, v in partitions.items()} != cfg['expected_partition_tracks'] or len(known) != 15:
        raise ValueError('Registered legal Train universe differs')
    policy_cfg = json.loads((ROOT / 'configs/trackocd_core/persistent_policy_training.json').read_text())
    receipt_path = historical / 'policy/training_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    fits = {(r['representation'], r['seed'], r['model']): r for r in receipt['fits']}
    weights_count = Counter(by[r['key']]['category_id'] for r in partitions['policy_train'])
    weights = {c: len(partitions['policy_train']) / (len(weights_count) * n) for c, n in weights_count.items()}
    out.mkdir(parents=True)
    results, geometry, gradients, candidate_rows, action_rows = [], [], [], [], []
    lineage, trace_writer = [], None
    raw_banks = {part: evidence_bank(cache, rows) for part, rows in partitions.items()}
    try:
        for representation in cfg['geometry_representations']:
            seeds = [None] if representation == 'A0_RAW' else cfg['seeds']
            for rep_seed in seeds:
                guard()
                model, rep_lineage = frozen_model(ROOT, representation, rep_seed)
                lineage.append({'representation': representation, 'seed': rep_seed, 'lineage': rep_lineage})
                proto = prototypes(cache, labels, known, model)
                banks = raw_banks if model is None else {part: evidence_bank(cache, rows, model) for part, rows in partitions.items()}
                for part, rows in partitions.items():
                    annotated = [{**r, **by[r['key']]} for r in rows]
                    for prefix in cfg['prefixes']:
                        vectors = np.stack([banks[part][prefix][r['key']]['embedding'] for r in rows])
                        raw = np.stack([raw_banks[part][prefix][r['key']]['embedding'] for r in rows])
                        geometry.append({'representation': representation, 'seed': rep_seed, 'partition': part,
                                         'prefix': prefix, **geometry_diagnostics(vectors, annotated, raw)})
                if representation == 'A2_CAPACITY_CONTROL':
                    continue
                for seed in cfg['seeds'] if representation == 'A0_RAW' else [rep_seed]:
                    for policy in cfg['policies']:
                        fit = fits[(representation, seed, policy)]
                        cp = fit['selected_checkpoint']['checkpoint']
                        path = ROOT / cp['path']
                        if path.stat().st_size != cp['bytes'] or sha256_file(path) != cp['sha256']:
                            raise ValueError('Policy checkpoint differs')
                        saved = torch.load(path, map_location='cpu', weights_only=True)
                        predictor = DecisionMLP().eval().requires_grad_(False)
                        predictor.load_state_dict(saved['state_dict'], strict=True)
                        bias = fit['operating_points']['1.0']['wait_bias']
                        for part, rows in partitions.items():
                            orders = registered_orders(sorted({r['video_id'] for r in rows}))
                            for name, order in orders.items():
                                for prefix in cfg['prefixes']:
                                    guard()
                                    key = {'representation': representation, 'policy': policy, 'seed': seed,
                                           'partition': part, 'order': name, 'prefix': prefix}
                                    sealed, diag, trace = diagnostic_replay(rows, order, prefix, proto, banks[part][prefix],
                                        predictor, by, policy_cfg['risk_constants'], bias, keep_trace=True)
                                    results.append({**key, 'wait_bias': bias, **diag, 'evaluation': evaluate(sealed, rows, labels)})
                                    for role in ('known', 'pseudo_novel'):
                                        counts = diag['candidate_counts_same_real_history']
                                        for topk in cfg['candidate_topk']:
                                            candidate_rows.append({**key, 'role': role, 'topk': topk,
                                                'queries': counts.get(f'{role}_queries', 0),
                                                'correct_available': counts.get(f'{role}_correct_available', 0),
                                                'correct_in_topk': counts.get(f'{role}_top{topk}_hit', 0)})
                                    for action in ('KNOWN', 'EXISTING', 'NEW', 'WAIT'):
                                        action_rows.append({**key, 'action': action, 'count': diag['counts'].get(action, 0)})
                                    private_rows = [{**key, **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in t.items()}}
                                                    for t in trace]
                                    table = pa.Table.from_pylist(private_rows)
                                    if trace_writer is None:
                                        # Nullable ranks may be absent in the first trace: pin types.
                                        schema = pa.schema([(f.name, pa.int64() if f.name in ('correct_known_rank',
                                            'correct_pure_cross_video_anonymous_rank', 'delayed_credit_sequence') else f.type)
                                            for f in table.schema])
                                        trace_writer = pq.ParquetWriter(out / 'private_traces.parquet', schema, compression='zstd')
                                    trace_writer.write_table(table.cast(trace_writer.schema))
                                    if representation == 'A2_EVIDENCE' and policy == 'D2_RISK_AWARE':
                                        s, d, _ = diagnostic_replay(rows, order, prefix, proto, banks[part][prefix], predictor,
                                                                   by, policy_cfg['risk_constants'], 0.)
                                        results.append({**key, 'wait_bias': 0., 'diagnostic_condition': 'ZERO_WAIT_BIAS',
                                                        **d, 'evaluation': evaluate(s, rows, labels)})
                                        if name == 'main':
                                            for intervention in cfg['oracle_interventions']:
                                                s, d, _ = diagnostic_replay(rows, order, prefix, proto, banks[part][prefix],
                                                    predictor, by, policy_cfg['risk_constants'], bias, intervention)
                                                results.append({**key, 'wait_bias': bias, **d, 'evaluation': evaluate(s, rows, labels)})
                            atomic_json(out / 'progress.json', {'status': 'RUNNING_CHECKPOINT_DIAGNOSTICS',
                                       'completed_cases': len(results), **key})
                        routes = partitions[cfg['gradient_partition']]
                        gradients.append({'representation': representation, 'seed': seed, 'policy': policy,
                            'original_training_active_terms': ['nominal_CE'] if policy == 'D1_SIMPLE_MLP' else
                                ['nominal_CE', 'expected_cost', 'delayed_credit'],
                            'D1_cost_gradients_are_hypothetical_not_original_contributions': policy == 'D1_SIMPLE_MLP',
                            'checkpoint': cp, **checkpoint_gradient_replay(predictor, routes,
                                sorted({r['video_id'] for r in routes}), cfg['gradient_prefix'], proto,
                                banks[cfg['gradient_partition']][cfg['gradient_prefix']], by,
                                policy_cfg['risk_constants'], weights, cfg['gradient_wait_bias'],
                                policy_cfg['D2_expected_cost_weight'], policy_cfg['D2_delayed_credit_weight'])})
                        print(json.dumps({'completed_cases': len(results), 'representation': representation,
                                          'policy': policy, 'seed': seed}), flush=True)
                del model, banks
    finally:
        if trace_writer is not None:
            trace_writer.close()
    expected = cfg['actual_cases'] + cfg['zero_bias_FULL_diagnostic_cases'] + cfg['oracle_MAIN_FULL_cases']
    if len(results) != expected or len(geometry) != cfg['geometry_cases'] or len(gradients) != cfg['gradient_replays']:
        raise ValueError('Incomplete registered diagnostics')
    result = {'status': 'COMPLETE_TRAIN_CHECKPOINT_SUBSTEP_NOT_COMPLETE_CVPR0',
              'preregistration_commit': remote, 'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES},
              'feature_manifest_sha256': cfg['feature_manifest_sha256'],
              'policy_receipt_sha256': sha256_file(receipt_path), 'representation_lineage': lineage,
              'cases': results, 'geometry': geometry, 'checkpoint_gradient_replays': gradients,
              'label_partitions_read': cfg['allowed_label_partitions'], 'final_holdout_labels_read': False,
              'Val_or_Test_read': False, 'optimization_steps': 0,
              'private_traces': {'filename': 'private_traces.parquet', 'sha256': sha256_file(out / 'private_traces.parquet')},
              'resources': {'wall_seconds': time.monotonic() - started, 'cpu_workers': 1,
                            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'training_GPU_hours': 0},
              'pending_followup': cfg['pending_followup']}
    csv_write(out / 'candidate_oracle_topk.csv', candidate_rows)
    csv_write(out / 'seed_action_distribution.csv', action_rows)
    atomic_json(out / 'result.json', result)
    atomic_json(out / 'progress.json', {'status': result['status'], 'completed_cases': len(results)})
    print(json.dumps({'status': result['status'], 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    main()
