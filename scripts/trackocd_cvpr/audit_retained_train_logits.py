#!/usr/bin/env python3
"""Stream retained Train traces into aggregate logit/error distributions."""
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
from src.trackocd_cvpr.logit_forensics import LogitForensics
from scripts.trackocd_cvpr.audit_historical_full import csv_write
SOURCES = ('configs/trackocd_cvpr/cvpr0_logit_forensics.json', 'src/trackocd_cvpr/logit_forensics.py',
           'scripts/trackocd_cvpr/audit_retained_train_logits.py')
KEYS = ('representation', 'policy', 'seed', 'partition', 'order', 'prefix')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preregistration-commit', required=True)
    p.add_argument('--git-proxy', default='')
    args = p.parse_args()
    cfg = json.loads((ROOT / SOURCES[0]).read_text())
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    git = ['git'] + (['-c', f'http.proxy={args.git_proxy}'] if args.git_proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote = subprocess.check_output(git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"],
                                     timeout=35, text=True, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()[0]
    if remote != args.preregistration_commit:
        raise ValueError('Exact remote aggregation registration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{remote}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered forensic source differs')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve real forensic attempts')
    started = time.monotonic()
    def guard():
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable'] < mem['MemTotal'] * safety['system_ram_reserve_fraction']:
            raise RuntimeError('RAM reserve')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg['rss_bytes_ceiling']:
            raise RuntimeError('RSS ceiling')
        if time.monotonic() - started > cfg['wall_seconds_ceiling'] or shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes']:
            raise RuntimeError('Wall/disk bound')
    guard()
    inp = ROOT / cfg['input_directory']
    if sha256_file(inp / 'result.json') != cfg['input_result_sha256']:
        raise ValueError('Registered completed diagnostic differs')
    source = json.loads((inp / 'result.json').read_text())
    if source['status'] != 'COMPLETE_TRAIN_CHECKPOINT_SUBSTEP_NOT_COMPLETE_CVPR0':
        raise ValueError('Only completed frozen Train trace')
    trace_path = inp / source['private_traces']['filename']
    if sha256_file(trace_path) != cfg['private_trace_sha256'] or cfg['private_trace_sha256'] != source['private_traces']['sha256']:
        raise ValueError('Retained trace hash differs')
    points = {tuple(c[k] for k in KEYS): c['wait_bias'] for c in source['cases']
              if c['intervention'] == 'ACTUAL_FROZEN_CHECKPOINT' and 'diagnostic_condition' not in c}
    if len(points) != cfg['expected_actual_cases']:
        raise ValueError('Actual case universe differs')
    import pyarrow.parquet as pq
    labels_path = ROOT / 'outputs/trackocd_core/features/train_first_v1/train_labels.parquet'
    role_rows = pq.read_table(labels_path, columns=['key', 'simulation_role'],
                              filters=[('partition', 'in', cfg['permitted_partitions'])]).to_pylist()
    roles = {r['key']: r['simulation_role'] for r in role_rows}
    accumulators = {}
    count = 0
    for batch in pq.ParquetFile(trace_path).iter_batches(batch_size=cfg['batch_rows'], use_threads=False):
        guard()
        for row in batch.to_pylist():
            key = tuple(row[k] for k in KEYS)
            if key not in points or row['physical_key'] not in roles:
                raise ValueError('Unregistered trace or role source')
            role = roles[row['physical_key']]
            a = accumulators.setdefault(key + (role,), LogitForensics())
            a.observe(row['logits'], row['features'], row['action'], role, row['correct_known_rank'],
                      row['correct_pure_cross_video_anonymous_rank'], row['costs'], row['nominal_target'],
                      row['actual_prefix'], points[key], row['sequence'], row['delayed_credit_sequence'])
            count += 1
    if count != cfg['expected_rows'] or len(accumulators) != cfg['expected_role_groups']:
        raise ValueError('Missing or duplicated trace universe')
    groups, logit_rows, error_rows = [], [], []
    for key, a in sorted(accumulators.items()):
        identity = dict(zip(KEYS + ('role',), key))
        r = a.finish()
        groups.append({**identity, **r})
        for channel, s in r['logit_channels'].items():
            logit_rows.append({**identity, 'channel': channel, **{k: v for k, v in s.items() if k != 'histogram'},
                              'fixed_histogram': json.dumps(s['histogram'], separators=(',', ':'))})
        for error, n in r['counts'].items():
            error_rows.append({**identity, 'diagnostic': error, 'count': n})
    out.mkdir(parents=True)
    csv_write(out / 'action_logits_distribution.csv', logit_rows)
    csv_write(out / 'candidate_vs_controller_errors.csv', error_rows)
    result = {'status': 'COMPLETE_RETAINED_TRAIN_LOGIT_FORENSICS_NOT_COMPLETE_CVPR0',
        'preregistration_commit': remote, 'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES},
        'input_result_sha256': cfg['input_result_sha256'], 'input_trace_sha256': cfg['private_trace_sha256'],
        'rows': count, 'groups': groups, 'raw_private_rows_published': False,
        'correct_top8_not_mislabeled_as_exposed_top1': True,
        'model_executed': False, 'optimization_steps': 0, 'Val_or_Test_read': False, 'final_holdout_labels_read': False,
        'resources': {'wall_seconds': time.monotonic() - started, 'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      'GPU_hours': 0},
        'pending': ['per-token causal memory trajectory', 'matched Train predicted-vs-GT inputs',
                    'premature-commit causal counterfactual beyond independent histories']}
    atomic_json(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 'rows': count, 'role_groups': len(groups), 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    main()
