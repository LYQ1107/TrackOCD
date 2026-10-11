#!/usr/bin/env python3
"""Finite unchanged-checkpoint Train replay with private per-state audit rows."""
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
SOURCES = ('configs/trackocd_cvpr/autonomous_research.json',
           'configs/trackocd_cvpr/cvpr0_state_trajectories.json',
           'src/trackocd_cvpr/state_trajectories.py',
           'src/trackocd_cvpr/checkpoint_diagnostics.py',
           'scripts/trackocd_cvpr/audit_train_state_trajectories.py')
KEYS = ('representation', 'policy', 'seed', 'partition', 'order', 'prefix')


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
        raise ValueError('Exact remote registration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{remote}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered source differs')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve every real diagnostic attempt')
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.trackocd_core.train_first_cache import TrainFirstCache
    from src.trackocd_core.train_first_experiment import frozen_model, evidence_bank, prototypes, evaluate
    from src.trackocd_core.persistent_policy import DecisionMLP
    from src.trackocd_cvpr.state_trajectories import replay_state_trajectories
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()

    def guard():
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable'] < mem['MemTotal'] * safety['system_ram_reserve_fraction']:
            raise RuntimeError('RAM reserve')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg['rss_bytes_ceiling']:
            raise RuntimeError('RSS ceiling')
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
        raise ValueError('Frozen model registry differs')
    for name, digest in json.loads(freeze_path.read_text())['protected_sha256'].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError('Protected source/input differs')
    cache_root = ROOT / 'outputs/trackocd_core/features/train_first_v1'
    if sha256_file(cache_root / 'manifest.json') != cfg['feature_manifest_sha256']:
        raise ValueError('Feature registry differs')
    cache = TrainFirstCache(cache_root)
    source = pq.read_table(cache_root / 'train_labels.parquet',
                           filters=[('partition', 'in', cfg['allowed_label_partitions'])]).to_pylist()
    routes_by_key = {r['key']: r for r in cache.rows}
    labels = [{**r, 'video_id': routes_by_key[r['key']]['video_id']} for r in source]
    by = {r['key']: r for r in labels}
    known = sorted({r['category_id'] for r in labels if r['partition'] == 'representation_fit'})
    parts = {part: [r for r in cache.rows if r['key'] in by and by[r['key']]['partition'] == part]
             for part in cfg['partitions']}
    if {k: len(v) for k, v in parts.items()} != {'development': 258, 'policy_train': 229} or len(known) != 15:
        raise ValueError('Train universe differs')
    receipt = json.loads((historical / 'policy/training_receipt.json').read_text())
    fits = {(f['representation'], f['seed'], f['model']): f for f in receipt['fits']}
    constants = json.loads((ROOT / 'configs/trackocd_core/persistent_policy_training.json').read_text())['risk_constants']
    previous = ROOT / 'outputs/trackocd_cvpr/cvpr0/checkpoint_replay_v1/result.json'
    if sha256_file(previous) != 'f0b302430963da915f8841a4069f70a6b9aeef4100e0608d739f9f4b91fccfc5':
        raise ValueError('Completed reference changed')
    completed = json.loads(previous.read_text())
    retained_path = previous.parent / completed['private_traces']['filename']
    if sha256_file(retained_path) != completed['private_traces']['sha256']:
        raise ValueError('Private reference trace differs')
    reference = {tuple(c[k] for k in KEYS): c for c in completed['cases']
                 if c['intervention'] == 'ACTUAL_FROZEN_CHECKPOINT' and 'diagnostic_condition' not in c}
    snapshot = pa.struct([('member_count', pa.int64()), ('category_count', pa.int64()),
                          ('purity', pa.float64()), ('mixed', pa.bool_()), ('video_count', pa.int64())])
    schema = pa.schema([(k, pa.int64() if k in {'seed', 'prefix'} else pa.string()) for k in KEYS] + [
        ('sequence', pa.int64()), ('video_id', pa.int64()), ('physical_key', pa.string()),
        ('actual_prefix', pa.int64()), ('action', pa.string()), ('category_id_evaluator_only', pa.int64()),
        ('role_evaluator_only', pa.string()), ('nominal_target_evaluator_only', pa.int64()),
        ('delayed_credit_sequence_evaluator_only', pa.int64()), ('logits', pa.list_(pa.float64())),
        ('cues', pa.list_(pa.float64())), ('candidate_token', pa.string()), ('selected_token', pa.string()),
        ('selected_known_id', pa.int64()), ('candidate_before', snapshot), ('selected_before', snapshot),
        ('selected_after', snapshot), ('bad_anonymous_write', pa.bool_()),
        ('first_bad_write_sequence', pa.int64()), ('first_cross_category_conflict_sequence', pa.int64())])
    out.mkdir(parents=True)
    trace_path = out / 'private_state_trajectories.parquet'
    cases, rows, csv_rows = [], 0, []
    with pq.ParquetWriter(trace_path, schema, compression='zstd') as writer:
        for representation in cfg['representations']:
            for seed in cfg['seeds']:
                guard()
                rep, lineage = frozen_model(ROOT, representation, seed)
                proto = prototypes(cache, labels, known, rep)
                fit = fits[(representation, seed, cfg['policy'])]
                cp = fit['selected_checkpoint']['checkpoint']
                if sha256_file(ROOT / cp['path']) != cp['sha256']:
                    raise ValueError('Selected policy differs')
                model = DecisionMLP().eval().requires_grad_(False)
                model.load_state_dict(torch.load(ROOT / cp['path'], map_location='cpu', weights_only=True)['state_dict'])
                bias = fit['operating_points']['1.0']['wait_bias']
                for part, routes in parts.items():
                    bank = evidence_bank(cache, routes, rep)
                    order = sorted({r['video_id'] for r in routes})
                    for prefix in cfg['prefixes']:
                        guard()
                        identity = dict(zip(KEYS, (representation, cfg['policy'], seed, part, 'main', prefix)))
                        sealed, summary, trace = replay_state_trajectories(routes, order, prefix, proto,
                            bank[prefix], model, by, constants, bias)
                        old = reference[tuple(identity[k] for k in KEYS)]
                        evaluation = evaluate(sealed, routes, labels)
                        if evaluation != old['evaluation'] or summary['mixed_state_count'] != old['mixed_states']:
                            raise AssertionError('Evaluator parity failed')
                        # Exact action/logit/physical-order parity, not merely equal headline metrics.
                        found = pq.read_table(retained_path, use_threads=False,
                            filters=[(k, '=', identity[k]) for k in KEYS]).to_pylist()
                        if len(found) != len(trace):
                            raise AssertionError('Trace universe parity')
                        for a, b in zip(trace, found):
                            if (a['sequence'], a['physical_key'], a['logits']) != (b['sequence'], b['physical_key'], b['logits']):
                                raise AssertionError('Exact logits/order parity')
                            if a['action'] != ('KNOWN', 'EXISTING', 'NEW', 'WAIT')[b['action']]:
                                raise AssertionError('Exact action parity')
                        writer.write_table(pa.Table.from_pylist([{**identity, **t} for t in trace], schema=schema))
                        rows += len(trace)
                        for state in summary['states']:
                            csv_rows.append({**identity, **state})
                        cases.append({**identity, 'wait_bias': bias, 'representation_lineage': lineage,
                            'summary': summary, 'exact_retained_logits_actions_order_and_evaluator_parity': True})
                        atomic_json(out / 'progress.json', {'status': 'RUNNING_STATE_TRAJECTORIES',
                            'cases': len(cases), 'rows': rows})
    guard()
    if len(cases) != cfg['expected_cases'] or rows != cfg['expected_rows']:
        raise ValueError('Incomplete registered state replay')
    if csv_rows:
        csv_write(out / 'memory_first_error.csv', csv_rows)
    result = {'status': 'COMPLETE_TRAIN_STATE_TRAJECTORIES_NOT_COMPLETE_CVPR0',
        'preregistration_commit': remote, 'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES},
        'reference_result_sha256': sha256_file(previous), 'cases': cases, 'rows': rows,
        'private_trace': {'filename': trace_path.name, 'bytes': trace_path.stat().st_size, 'sha256': sha256_file(trace_path)},
        'raw_traces_published': False, 'optimization_steps': 0, 'GT_in_live_memory_or_forward': False,
        'Val_or_Test_access': False, 'final_holdout_labels_read': False,
        'resources': {'wall_seconds': time.monotonic() - started,
            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'training_GPU_hours': 0},
        'pending': ['matched lawful predicted Train input', 'fixed-history premature-commit counterfactual',
                    'new dense online identity-independent protocol']}
    atomic_json(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 'cases': len(cases), 'rows': rows, 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    main()
