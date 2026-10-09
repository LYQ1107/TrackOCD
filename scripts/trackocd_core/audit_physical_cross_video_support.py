#!/usr/bin/env python3
"""One CPU, existing sealed tracks only; no models/semantics/feature extraction."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import datetime as dt
import io
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, atomic_write_text, sha256_file
from src.trackocd_core.evaluation.physical_support import cross_video_support
from scripts.trackocd_core.audit_frontend_coverage import match_video
from scripts.trackocd_core.evaluate_masa_physical_qualification import read_projection
from scripts.trackocd_core.run_gt_pilot_baselines import registered_orders
from scripts.trackocd_core.audit_frontend import TRACK_ROOT

PUBLIC = ROOT / "configs/trackocd_core/physical_cross_video_support.json"
PRIVATE = ROOT / "outputs/trackocd_core/audit/physical_cross_video_support_private"
SUMMARY = ROOT / "outputs/trackocd_core/audit/physical_cross_video_support.json"


def guard(config):
    mem = {k: int(v.split()[0]) for k, v in (s.split(":", 1) for s in Path('/proc/meminfo').read_text().splitlines())}
    limit = config["limits"]
    if (mem['MemAvailable'] < mem['MemTotal'] * limit['minimum_ram_headroom_fraction']
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limit['host_rss_bytes']
            or shutil.disk_usage(ROOT).free < limit['minimum_disk_headroom_bytes']):
        raise RuntimeError("CPU-only audit resource boundary; no foreign process touched")


def main(commit):
    started = time.monotonic()
    if PRIVATE.exists() or SUMMARY.exists():
        raise ValueError("Preserve existing attempt/result; no automatic rematching")
    delivery = json.loads((ROOT / 'outputs/trackocd_core/audit/physical_cross_video_support_preregistration_delivery.json').read_text())
    if delivery['commit'] != commit or delivery['remote_verified'] is not True:
        raise ValueError("Missing exact remote-verified preregistration")
    for name in ('configs/trackocd_core/physical_cross_video_support.json', 'scripts/trackocd_core/audit_physical_cross_video_support.py',
                 'src/trackocd_core/evaluation/physical_support.py', 'src/trackocd_core/evaluation/persistent.py'):
        if subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError("Audit source/config changed after preregistration")
    config = json.loads(PUBLIC.read_text())
    if delivery['config_sha256'] != sha256_file(PUBLIC):
        raise ValueError("Preregistered config identity mismatch")
    for name, expected in config['unchanged_helpers'].items():
        if sha256_file(ROOT / name) != expected: raise ValueError('Frozen geometry helper changed')
    paths = {k: Path(config[k]) if Path(config[k]).is_absolute() else ROOT / config[k]
             for k in ('full_physical_receipt', 'private_metadata_plan', 'annotation', 'roles')}
    for name, path in paths.items():
        if sha256_file(path) != config[name + '_sha256']: raise ValueError('Pinned input changed: ' + name)
    if sha256_file(ROOT / 'configs/trackocd_core/evaluation_protocol.json') != config['evaluation_protocol_sha256']:
        raise ValueError('Fixed evaluator protocol changed')
    roots = [ROOT, Path('/data3/liuyeqiang/.venvs/trackocd-masa-smoke')]
    storage = sum(int(s.split()[0]) for s in subprocess.check_output(['du', '-s', '-B1', *map(str, roots)], text=True).splitlines())
    if storage + config['limits']['new_output_allocated_bytes'] > config['limits']['overall_soft_bytes']:
        raise RuntimeError('Conservative repository+owned-env exceeds soft storage budget')
    guard(config)
    planned_mem = {k: int(v.split()[0]) for k, v in (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
    if planned_mem['MemAvailable'] * 1024 - config['limits']['host_rss_bytes'] < planned_mem['MemTotal'] * 1024 * .25:
        raise RuntimeError('Working-memory plan would cross 25% system headroom')
    def barrier(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto', 'subprocess.Popen', 'os.system'}:
            raise PermissionError('CPU audit prohibits network/child execution')
        if event == 'open' and isinstance(args[0], (str, bytes)) and '/TAO-Amodal/' in os.fsdecode(args[0]):
            path = os.fsdecode(args[0])
            if '/test' in path or '/train' in path or '/frames/' in path:
                raise PermissionError('Only fixed Val annotation, no Train/Test/images in CPU audit')
    sys.addaudithook(barrier)
    PRIVATE.mkdir(parents=True, exist_ok=False)
    plan = json.loads(paths['private_metadata_plan'].read_text())
    prior = json.loads(paths['full_physical_receipt'].read_text())
    gt = json.loads(paths['annotation'].read_text()); roles = json.loads(paths['roles'].read_text())
    known, novel = set(roles['known_ids']), set(roles['novel_ids'])
    videos = [v['video_id'] for v in plan['videos']]
    if len(videos) != config['videos'] or sum(len(v['images']) for v in plan['videos']) != config['images']:
        raise ValueError('Not full fixed universe')
    grouped = defaultdict(dict)
    for ann in gt['annotations']:
        category, vid, local = int(ann['category_id']), int(ann['video_id']), int(ann['track_id'])
        if category not in known | novel: continue  # Same original headline universe; no prediction filtering.
        key = f'{vid}_{local}'
        row = grouped[vid].setdefault(key, {'key': key, 'category': category, 'role': 'known' if category in known else 'novel', 'gt_local_id': local, 'boxes': {}})
        if row['category'] != category: raise ValueError('Inconsistent GT category')
        x, y, w, h = map(float, ann['bbox']); row['boxes'][int(ann['image_id'])] = [x, y, x+w, y+h]
    records = {name: [] for name in config['frontends']}
    prior_by_video = {n: {r['video_id']: r for r in prior['results'][n]['per_video']} for n in records}
    run = ROOT / 'outputs/trackocd_core/physical/masa_full_val_annotated'
    sealed = {r['video_id']: r for r in json.loads((run / 'prediction_manifest.json').read_text())['videos']}
    for position, video in enumerate(plan['videos']):
        guard(config); vid = video['video_id']; targets = list(grouped[vid].values())
        for name in records:
            source = run / 'shards' / sealed[vid]['npz_filename'] if name == 'MASA_NATIVE' else TRACK_ROOT / f'video_{vid:04d}.npz'
            expected = prior_by_video[name][vid]
            if source.stat().st_size != expected['source_npz_bytes'] or sha256_file(source) != expected['source_npz_sha256']:
                raise ValueError('Changed physical NPZ; do not rerun or repair')
            frames = read_projection(source, video['images'])
            matches = match_video(targets, {i: (f[0], f[1]) for i, f in frames.items()})['matches']
            ids, lengths = np.unique(np.concatenate([f[0] for f in frames.values()]), return_counts=True)
            length_by_id = dict(zip(ids.tolist(), lengths.tolist()))
            for role in ('known', 'novel'):
                eligible = [r for r in targets if r['role'] == role]
                covered = sum(matches.get(r['key'], {}).get('reliable', False) for r in eligible)
                if len(eligible) != expected['coverage'][role]['gt_tracks'] or covered != expected['coverage'][role]['reliably_observed']:
                    raise ValueError('Per-video coverage differs from sealed full audit')
            for target in targets:
                matched = matches.get(target['key'], {}); pred = matched.get('predicted_local_track_id') if matched.get('reliable') else None
                records[name].append({'video_id': vid, 'gt_local_id': target['gt_local_id'], 'category_id': target['category'], 'role': target['role'],
                                      'reliable_predicted_local_id': pred, 'reliable_predicted_observations': int(length_by_id[pred]) if pred is not None else 0,
                                      'assigned_temporal_iou': matched.get('temporal_iou')})
            del frames
        atomic_json(PRIVATE / 'progress.json', {'status': 'EVALUATOR_ONLY_JOIN_IN_PROGRESS', 'completed_videos': position+1, 'total_videos': len(videos), 'elapsed_seconds': time.monotonic()-started})
        if (position+1) % 100 == 0: print(json.dumps({'joined_videos': position+1}), flush=True)
    orders = registered_orders(videos)
    if list(orders) != config['registered_orders']: raise ValueError('Wrong registered orders')
    result = {'schema_version': 'trackocd.core.physical_cross_video_support_result.v1', 'status': 'COMPLETE_POSTHOC_PHYSICAL_SUPPORT_NOT_SEMANTIC_SUCCESS',
              'scope': config['scope'], 'preregistration_commit': commit, 'config_sha256': sha256_file(PUBLIC), 'full_physical_receipt_sha256': config['full_physical_receipt_sha256'],
              'videos': len(videos), 'images': config['images'], 'frontends': {}, 'model_inference_or_training': False, 'test_access': False,
              'primary_freeze_permitted': False, 'semantic_scientific_pass_permitted': False, 'gt_or_geometry_feedback_to_model': False,
              'short_tracks_removed_from_denominator': False, 'support_boundary': config['support_definition'], 'prefix_boundary': config['short_track_boundary']}
    table = io.StringIO(); writer = csv.writer(table)
    writer.writerow(['scope','frontend','order','prefix','fixed_gt_reuse_opportunities','reliable_current','both_sides_at_least_p','fraction','supported_categories'])
    for name, rows in records.items():
        coverage = {role: {'gt_tracks': sum(r['role']==role for r in rows), 'reliably_observed': sum(r['role']==role and r['reliable_predicted_observations']>0 for r in rows)} for role in ('known','novel')}
        if coverage['known']['gt_tracks'] != config['known_gt_tracks'] or coverage['novel']['gt_tracks'] != config['novel_gt_tracks']: raise ValueError('Changed fixed target universe')
        replay = {order: cross_video_support(rows, sequence, config['prefixes']) for order, sequence in orders.items()}
        result['frontends'][name] = {'coverage_reproduces_frozen_audit': True, 'coverage': coverage, 'orders': replay,
            'prefix_four_order_mean_std': {str(p): {'mean': float(np.mean([v['prefix_support'][str(p)]['fraction_of_all_fixed_gt_opportunities'] for v in replay.values()])),
                                                   'std_ddof0': float(np.std([v['prefix_support'][str(p)]['fraction_of_all_fixed_gt_opportunities'] for v in replay.values()]))} for p in config['prefixes']}}
        for order, row in replay.items():
            for p in config['prefixes']:
                d = row['prefix_support'][str(p)]
                writer.writerow(['M1_PHYSICAL_SUPPORT_NOT_COMMIT_CT', name, order, p, row['fixed_gt_reuse_opportunities'], row['reliable_current_opportunities'], d['both_earlier_and_current_have_at_least_p'], d['fraction_of_all_fixed_gt_opportunities'], d['categories_with_supported_pairs']])
    private_join = PRIVATE / 'evaluator_only_geometry_join.json'
    atomic_json(private_join, {'evaluator_only_not_model_input': True, 'config_sha256': result['config_sha256'], 'frontends': records})
    result['private_join'] = {'bytes': private_join.stat().st_size, 'sha256': sha256_file(private_join), 'published': False}
    guard(config)
    allocated = sum(p.stat().st_blocks*512 for p in PRIVATE.rglob('*') if p.is_file())
    if allocated > config['limits']['new_output_allocated_bytes']: raise RuntimeError('Preserve oversized output, no complete publication')
    result['resources'] = {'cpu_workers': 1, 'gpu_used': False, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                           'wall_seconds': time.monotonic()-started, 'conservative_repo_and_owned_env_allocated_bytes_before': storage, 'private_allocated_bytes_before_marker': allocated}
    result['completed_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(PRIVATE / '.done', {'status': result['status'], 'private_join_sha256': result['private_join']['sha256'], 'config_sha256': result['config_sha256']})
    atomic_json(SUMMARY, result)
    atomic_write_text(ROOT / 'outputs/trackocd_core/tables/m1_physical_cross_video_support.csv', table.getvalue())
    print(json.dumps({'status': result['status'], 'resources': result['resources']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--preregistration-commit', required=True)
    main(parser.parse_args().preregistration_commit)
