#!/usr/bin/env python3
"""Registered Train-only memory/known-count and actual-ID-renumber audit."""
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
SOURCES = ('configs/trackocd_cvpr/autonomous_research.json', 'configs/trackocd_cvpr/cvpr0_memory_controls.json',
           'src/trackocd_cvpr/memory_controls.py', 'src/trackocd_cvpr/checkpoint_diagnostics.py',
           'scripts/trackocd_cvpr/audit_memory_controls.py')


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
        raise ValueError('Preserve existing attempts')
    import torch
    import pyarrow.parquet as pq
    from src.trackocd_core.train_first_cache import TrainFirstCache
    from src.trackocd_core.train_first_experiment import frozen_model, evidence_bank, prototypes, evaluate
    from src.trackocd_core.train_first_replay import replay
    from src.trackocd_core.persistent_policy import DecisionMLP
    from src.trackocd_cvpr.memory_controls import controlled_replay, prototype_subset, renumber_routes, compare_renumbered
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
            raise RuntimeError('Free disk reserve')

    guard()
    historical = ROOT / 'outputs/trackocd_core/core_training/train_first_v1'
    freeze_path = historical / 'model_freeze.json'
    if sha256_file(freeze_path) != cfg['model_freeze_sha256']:
        raise ValueError('Frozen model registry differs')
    for name, digest in json.loads(freeze_path.read_text())['protected_sha256'].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError('Protected input differs')
    cache_root = ROOT / 'outputs/trackocd_core/features/train_first_v1'
    if sha256_file(cache_root / 'manifest.json') != cfg['feature_manifest_sha256']:
        raise ValueError('Frozen features differ')
    cache = TrainFirstCache(cache_root)
    source = pq.read_table(cache_root / 'train_labels.parquet',
                           filters=[('partition', 'in', cfg['allowed_label_partitions'])]).to_pylist()
    route_map = {r['key']: r for r in cache.rows}
    labels = [{**r, 'video_id': route_map[r['key']]['video_id']} for r in source]
    by = {r['key']: r for r in labels}
    known = sorted({r['category_id'] for r in labels if r['partition'] == 'representation_fit'})
    parts = {p: [r for r in cache.rows if r['key'] in by and by[r['key']]['partition'] == p] for p in cfg['partitions']}
    if {k: len(v) for k, v in parts.items()} != {'development': 258, 'policy_train': 229}:
        raise ValueError('Registered Train universe differs')
    receipt = json.loads((historical / 'policy/training_receipt.json').read_text())
    fits = {(f['representation'], f['seed'], f['model']): f for f in receipt['fits']}
    out.mkdir(parents=True)
    capacity_cases, rename_cases, baselines = [], [], []
    for representation in cfg['renumber_representations']:
        for seed in cfg['seeds']:
            guard()
            rep, lineage = frozen_model(ROOT, representation, seed)
            proto = prototypes(cache, labels, known, rep)
            fit = fits[(representation, seed, cfg['renumber_policy'])]
            cp = fit['selected_checkpoint']['checkpoint']
            if sha256_file(ROOT / cp['path']) != cp['sha256']:
                raise ValueError('Policy checkpoint differs')
            model = DecisionMLP().eval().requires_grad_(False)
            model.load_state_dict(torch.load(ROOT / cp['path'], map_location='cpu', weights_only=True)['state_dict'])
            bias = fit['operating_points']['1.0']['wait_bias']
            for part, rows in parts.items():
                bank = evidence_bank(cache, rows, rep)
                order = sorted({r['video_id'] for r in rows})
                altered = renumber_routes(rows)
                for prefix in cfg['renumber_prefixes']:
                    guard()
                    a, ar = controlled_replay(rows, order, prefix, proto, bank[prefix], model, bias, None)
                    b, br = controlled_replay(altered, order, prefix, proto, bank[prefix], model, bias, None)
                    rename_cases.append({'representation': representation, 'seed': seed, 'partition': part, 'prefix': prefix,
                        **compare_renumbered(a, b, rows, altered, order, prefix),
                        'original': {'runtime': ar, 'evaluation': evaluate(a, rows, labels)},
                        'renumbered': {'runtime': br, 'evaluation': evaluate(b, altered, labels)}})
                    if representation == 'A0_RAW' and seed == cfg['seeds'][0]:
                        a, _ = replay(rows, order, prefix, proto, cfg['fixed_baseline_thresholds'], bank[prefix])
                        b, _ = replay(altered, order, prefix, proto, cfg['fixed_baseline_thresholds'], bank[prefix])
                        baselines.append({'representation': 'A0_RAW', 'backend': 'FIXED_B1_DIAGNOSTIC',
                            'partition': part, 'prefix': prefix,
                            **compare_renumbered(a, b, rows, altered, order, prefix),
                            'original': evaluate(a, rows, labels), 'renumbered': evaluate(b, altered, labels)})
                if representation == cfg['capacity_representation']:
                    for count in cfg['known_counts']:
                        subset = prototype_subset(proto, count)
                        for capacity in cfg['capacities']:
                            guard()
                            sealed, runtime = controlled_replay(rows, order, cfg['capacity_prefix'], subset,
                                bank[cfg['capacity_prefix']], model, bias, capacity)
                            capacity_cases.append({'representation': representation, 'policy': cfg['capacity_policy'],
                                'seed': seed, 'partition': part, 'prefix': cfg['capacity_prefix'],
                                'known_count': count, 'capacity': capacity, 'runtime': runtime,
                                'full_original_label_denominators_retained': True,
                                'evaluation': evaluate(sealed, rows, labels)})
            atomic_json(out / 'progress.json', {'status': 'RUNNING_CONTROLS', 'renumber_pairs': len(rename_cases),
                                               'capacity_cases': len(capacity_cases), 'representation': representation, 'seed': seed})
    if (len(capacity_cases), len(rename_cases), len(baselines)) != (
            cfg['capacity_cases'], cfg['renumber_pairs'], cfg['baseline_renumber_pairs']):
        raise ValueError('Incomplete controls')
    result = {'status': 'COMPLETE_MEMORY_CONTROLS_NOT_COMPLETE_CVPR0', 'preregistration_commit': remote,
        'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES},
        'capacity_cases': capacity_cases, 'renumber_cases': rename_cases, 'baseline_renumber_cases': baselines,
        'original_denominators_not_filtered_by_known_subset': True, 'GT_in_visual_memory_or_step': False,
        'Val_or_Test_access': False, 'final_holdout_labels_read': False, 'optimization_steps': 0,
        'resources': {'wall_seconds': time.monotonic() - started,
            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'training_GPU_hours': 0},
        'pending_CVPR0': ['lawful matched Train predicted-track input comparison', 'synthesis and mechanism preregistration']}
    atomic_json(out / 'result.json', result)
    atomic_json(out / 'progress.json', {'status': result['status']})
    print(json.dumps({'status': result['status'], 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    main()
