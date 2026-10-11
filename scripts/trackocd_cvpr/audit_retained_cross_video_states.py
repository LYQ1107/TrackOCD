#!/usr/bin/env python3
"""No-model finite posthoc of retained private state/video chronology."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.cross_video_forensics import CrossVideoTrajectories
from scripts.trackocd_cvpr.audit_historical_full import csv_write
SOURCES = ('configs/trackocd_cvpr/cvpr0_cross_video_forensics.json',
           'src/trackocd_cvpr/cross_video_forensics.py',
           'scripts/trackocd_cvpr/audit_retained_cross_video_states.py')
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
        text=True, timeout=35, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()[0]
    if remote != args.preregistration_commit:
        raise ValueError('Exact remote registration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{remote}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered aggregation source differs')
    out, inp = ROOT / cfg['output_directory'], ROOT / cfg['input_directory']
    if out.exists():
        raise ValueError('Preserve existing aggregation')
    started = time.monotonic()
    if sha256_file(inp / 'result.json') != cfg['input_result_sha256']:
        raise ValueError('Registered completed state result differs')
    completed = json.loads((inp / 'result.json').read_text())
    if completed['status'] != 'COMPLETE_TRAIN_STATE_TRAJECTORIES_NOT_COMPLETE_CVPR0':
        raise ValueError('Only completed state trajectories')
    trace = inp / completed['private_trace']['filename']
    if sha256_file(trace) != cfg['private_trace_sha256']:
        raise ValueError('Private chronology changed')
    expected = {tuple(c[k] for k in KEYS): c for c in completed['cases']}
    import pyarrow.parquet as pq
    accumulators, rows = {}, 0
    for batch in pq.ParquetFile(trace).iter_batches(batch_size=1024, use_threads=False):
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']:
            raise RuntimeError('RAM reserve')
        if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg['rss_bytes_ceiling']
                or time.monotonic() - started > cfg['wall_seconds_ceiling']):
            raise RuntimeError('RSS/wall bound')
        for row in batch.to_pylist():
            key = tuple(row[k] for k in KEYS)
            if key not in expected:
                raise ValueError('Unregistered case')
            accumulators.setdefault(key, CrossVideoTrajectories()).consume(row)
            rows += 1
    results = []
    for key, a in accumulators.items():
        states = a.finish()
        original = expected[key]['summary']['states']
        if len(states) != len(original):
            raise ValueError('State universe differs')
        for s, old in zip(states, original):
            if any(s[k] != old[k] for k in ('token', 'member_count', 'category_count', 'purity', 'mixed', 'video_count')):
                raise ValueError('Final state parity')
            results.append({**dict(zip(KEYS, key)), **s})
    if (len(accumulators), rows, len(results)) != (cfg['expected_cases'], cfg['expected_rows'], cfg['expected_states']):
        raise ValueError('Incomplete posthoc universe')
    out.mkdir(parents=True)
    csv_write(out / 'cross_video_state_reuse.csv', results)
    result = {'status': 'COMPLETE_RETAINED_CROSS_VIDEO_FORENSICS_NOT_COMPLETE_CVPR0',
        'preregistration_commit': remote, 'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES},
        'input_result_sha256': cfg['input_result_sha256'], 'input_trace_sha256': cfg['private_trace_sha256'],
        'cases': len(accumulators), 'rows': rows, 'states': results,
        'prior_and_post_purity_membership_and_final_state_parity': True,
        'new_model_forward': False, 'new_GT_labels_read': False, 'raw_private_rows_published': False,
        'resources': {'wall_seconds': time.monotonic() - started,
            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'training_GPU_hours': 0},
        'interpretation': cfg['interpretation']}
    atomic_json(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 'rows': rows, 'states': len(results), 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    main()
