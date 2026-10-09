#!/usr/bin/env python3
"""Read-only asset/protocol audit. Never opens TAO Test annotations.

Only the explicitly named Train/Validation annotation files are parsed.
Generated audit evidence is written in this task's new output namespace.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file

DATASET = Path('/data3/liuyeqiang/TAO-Amodal')
PANDAS = Path('/data3/liuyeqiang/pandas_bytetrack_tao_val')


def command(args: list[str]) -> dict:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=15)
    return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip()}


def read_annotation(path: Path, split: str) -> dict:
    expected = {'train': 'train.json', 'val': 'validation.json'}
    if split not in expected or path.name != expected[split]:
        raise ValueError('TAO_TEST_OR_UNREGISTERED_ANNOTATION_FORBIDDEN')
    if path.resolve() != (DATASET / 'annotations' / expected[split]).resolve():
        raise ValueError('UNREGISTERED_ANNOTATION_PATH')
    with path.open() as handle:
        return json.load(handle)


def validate_roles(roles: dict) -> None:
    sets = [set(roles[key]) for key in ('known_ids', 'novel_ids', 'distractor_ids')]
    if any(len(s) != len(roles[k]) for s, k in zip(sets, ('known_ids', 'novel_ids', 'distractor_ids'))):
        raise ValueError('duplicate role ID')
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('overlapping roles')


def file_record(path: Path, *, hash_file: bool = True) -> dict:
    record = {'path': str(path), 'is_symlink': path.is_symlink(),
              'resolved_path': str(path.resolve()), 'exists': path.exists()}
    if path.is_file():
        stat = path.stat()
        record.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                      sha256=sha256_file(path) if hash_file else None)
    return record


def annotation_summary(data: dict, split: str, roles: dict) -> dict:
    known = set(roles['known_ids'])
    novel = set(roles['novel_ids'])
    distractor = set(roles['distractor_ids'])
    by_track = {}
    per_category = defaultdict(lambda: {'tracks': 0, 'videos': set(), 'observations': 0})
    role_counts = Counter()
    for ann in data['annotations']:
        category = int(ann['category_id'])
        key = (int(ann['video_id']), int(ann['track_id']))
        if key in by_track and by_track[key] != category:
            raise ValueError(f'inconsistent track category {key}')
        if key not in by_track:
            per_category[category]['tracks'] += 1
        by_track[key] = category
        per_category[category]['videos'].add(key[0])
        per_category[category]['observations'] += 1
        role = ('old' if category in known else 'new' if category in novel else
                'distractor' if category in distractor else 'train_unassigned')
        if split == 'val' and role == 'train_unassigned':
            raise ValueError('Validation category outside inherited roles')
        role_counts[role] += 1
    supported = known & per_category.keys()
    image_paths = [DATASET / 'frames' / im['file_name'] for im in data['images']]
    missing_frames = [str(p) for p in image_paths if not p.is_file()]
    return {
        'videos': len(data['videos']), 'annotated_images': len(data['images']),
        'annotations': len(data['annotations']), 'physical_gt_tracks': len(by_track),
        'observed_categories': len(per_category), 'role_annotation_counts': dict(role_counts),
        'missing_annotated_frame_count': len(missing_frames),
        'missing_annotated_frame_examples': missing_frames[:10],
        'train_supported_known_ids': sorted(supported) if split == 'train' else None,
        'train_zero_shot_known_ids': sorted(known - supported) if split == 'train' else None,
        'train_known_category_stats': {
            str(c): {'tracks': per_category[c]['tracks'],
                     'video_count': len(per_category[c]['videos']),
                     'observations': per_category[c]['observations']}
            for c in sorted(supported)
        } if split == 'train' else None,
        'model_training_annotation_allowlist': 'Train categories in known_ids ONLY',
        'other_train_categories_training_permitted': False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/trackocd_core/audit/assets.json')
    args = parser.parse_args()
    roles_path = ROOT / 'configs/trackocd_core/roles.json'
    roles = json.loads(roles_path.read_text())
    validate_roles(roles)
    recovered_files = {}
    for role_name, filename in (('known_base', 'known_ids.json'), ('genuine_novel', 'unknown_ids_val.json'), ('distractor', 'distractor_ids.json')):
        path = ROOT / 'outputs/trackocd_core/recovered_splits' / filename
        if not path.is_file() or sha256_file(path) != roles['original_files'][role_name]['sha256']:
            raise ValueError(f'RECOVERED_ROLE_ASSET_MISSING_OR_CORRUPT: {filename}')
        recovered_files[filename] = file_record(path)
    official = PANDAS / 'vendor/Open-World-Tracking'
    coco_map = official / 'datasets/coco_id2tao_id.json'
    distractor_map = official / 'datasets/distractor_classes.json'
    official_known = set(map(int, json.loads(coco_map.read_text()).values()))
    distractor_groups = json.loads(distractor_map.read_text())
    official_distractor = {int(value) for values in distractor_groups.values() for value in values}
    if official_known != set(roles['known_ids']) or official_distractor != set(roles['distractor_ids']):
        raise ValueError('inherited roles disagree with pinned official maps')
    annotations = {}
    for split, name in (('train', 'train.json'), ('val', 'validation.json')):
        path = DATASET / 'annotations' / name
        annotations[split] = dict(file_record(path), **annotation_summary(read_annotation(path, split), split, roles))
    source_paths = [
        'configs/trackocd_v2/protocol.json', 'src/trackocd_v2/schema.py',
        'src/trackocd_v2/protocol.py', 'src/trackocd_v2/methods/nearest_prototype.py',
        'src/trackocd_v2/methods/dpmeans.py', 'src/trackocd_v2/methods/phe_track.py',
        'src/trackocd_v2/evaluation/standard_ocd.py', 'src/trackocd_v2/evaluation/persistent.py',
        'scripts/trackocd_v2/run_gt_baselines.py', 'scripts/trackocd_v2/run_pred_baselines.py',
        'scripts/trackocd_v2/build_predicted_stream.py', 'scripts/trackocd_v2/build_common_features.py',
        'scripts/trackocd_v2/audit_geometry.py',
    ]
    assets = [
        ('gt_track_features', ROOT / 'data/caches/features/dinov2_vitb14', 'missing; do not replace with another encoder'),
        ('v2_gt_features', ROOT / 'outputs/trackocd_v2/features/gt_tracks', 'missing local recovered cache'),
        ('v2_predicted_features', ROOT / 'outputs/trackocd_v2/features/formal/covtrack_native', 'NAS shard final state unverified'),
        ('legacy_predicted_stream', ROOT / 'data/tao_ow_ocd_v1/public/pred_track_stream.jsonl', '649378-track historical stream not restored'),
        ('selected_frontend_state', ROOT / 'outputs/trackocd_v2/audit/autonomous_state.json', 'historical state not current state'),
        ('dinov2_checkpoint', ROOT / 'checkpoints/dinov2_vitb14_pretrain.pth', 'official source/hash in assets/downloads_manifest.json'),
        ('simowt_checkpoint', ROOT / 'checkpoints/simowt_weight.pth', 'official source/hash in assets/downloads_manifest.json'),
        ('phe_checkpoint', ROOT / 'runs/phe_track/dinov2_seed1027/checkpoint.pth', 'historical known coverage 48/78; representation compatibility must be checked'),
        ('pandas_frozen_tracks', PANDAS / 'outputs/score_fix_full/bytetrack/BT-FG-0', 'reference only; do not promote as reliable frontend without quality audit'),
    ]
    pandas_outputs = [file_record(p, hash_file=False) for p in sorted((PANDAS / 'outputs').iterdir())]
    v2_tree = command(['git', 'rev-parse', 'codex/trackocd-v2^{tree}'])['stdout']
    root_tree = command(['git', 'rev-parse', 'HEAD^{tree}'])['stdout']
    meminfo = {k: int(v.split()[0]) for k, v in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    stat = __import__('os').statvfs(ROOT)
    source_audit_path = ROOT / 'outputs/trackocd_core/audit/nas_source_audit.json'
    source_audit = json.loads(source_audit_path.read_text()) if source_audit_path.is_file() else None
    source_comparison = []
    for entry in source_audit['source_code']['inventory'] if source_audit else []:
        local = file_record(ROOT / entry['relative_path'])
        classification = ('MISSING_LOCAL' if not local['exists'] else
                          'DIFFERENT_BYTE_SIZE' if local.get('size_bytes') != entry['size_bytes'] else
                          'SAME_SIZE_NOT_BYTE_IDENTITY_PROOF')
        source_comparison.append({'relative_path': entry['relative_path'],
                                  'source_size_bytes': entry['size_bytes'],
                                  'source_sha256': None, 'local_file': local,
                                  'classification': classification})
    checkpoint_receipt_path = ROOT / 'outputs/trackocd_core/audit/dino_checkpoint_recovery.json'
    checkpoint_receipt = json.loads(checkpoint_receipt_path.read_text()) if checkpoint_receipt_path.is_file() else None
    dependency_receipt_path = ROOT / 'outputs/trackocd_core/audit/parquet_dependency_smoke.json'
    dependency_receipt = json.loads(dependency_receipt_path.read_text()) if dependency_receipt_path.is_file() else None
    audit = {
        'schema_version': 'trackocd.core.asset_audit.v1',
        'status': 'M0_AUDIT_COMPLETE_WITH_RECOVERY_GAPS',
        'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'git': {key: command(args_) for key, args_ in {
            'branch': ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            'head': ['git', 'rev-parse', 'HEAD'], 'status': ['git', 'status', '--porcelain=v1'],
            'branches': ['git', 'branch', '-avv'], 'worktrees': ['git', 'worktree', 'list', '--porcelain'],
            'stashes': ['git', 'stash', 'list', '--format=%gd %H %gs'],
        }.items()},
        'v2_source_tree': v2_tree, 'ancestor_delivery_tree': root_tree,
        'v2_source_files': [file_record(ROOT / p) for p in source_paths],
        'roles': {'path': str(roles_path), 'sha256': sha256_file(roles_path),
                  'counts': {k: len(roles[k]) for k in ('known_ids', 'novel_ids', 'distractor_ids')},
                  'inherited_ids_recovered_from_complete_machine_audit': True,
                  'original_split_bytes_recovered': True,
                  'recovered_original_files': recovered_files,
                  'official_known_and_distractor_content_equal': True,
                  'source': roles['source'], 'original_file_hashes': roles['original_files']},
        'official_role_sources': {'repository': 'https://github.com/YangLiu14/Open-World-Tracking',
                                  'git_head': command(['git', '-C', str(official), 'rev-parse', 'HEAD']),
                                  'known_map': file_record(coco_map),
                                  'distractor_map': file_record(distractor_map)},
        'annotations': annotations,
        'assets': [dict(file_record(p), kind=k, reuse_note=note) for k, p, note in assets],
        'existing_pandas_output_directories': pandas_outputs,
        'resource_snapshot': {'available_disk_bytes': stat.f_bavail * stat.f_frsize,
                              'mem_available_kib': meminfo['MemAvailable'],
                              'mem_total_kib': meminfo['MemTotal'],
                              'gpus': command(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv'])},
        'budget': {'soft_bytes': 15 * 2**30, 'hard_bytes': 30 * 2**30},
        'historical_cache_evidence': {'last_machine_observation_utc': '2026-09-17T18:28:14.579Z',
                                     'journal_line': 22062, 'atomic_shards': 65,
                                     'observations': 247756, 'status': 'RUNNING',
                                     'current_final_state_known': False, 'files_recovered': False},
        'current_nas_source_audit': {
            'evidence_file': file_record(source_audit_path),
            'source_turn': source_audit['source'] if source_audit else None,
            'source_code': {key: source_audit['source_code'][key] for key in
                            ('python_file_count', 'python_payload_bytes', 'source_bytes_restored_on_a100',
                             'exact_source_hashes_verified', 'source_git_head_verified')} if source_audit else None,
            'formal_cache': source_audit['formal_cache'] if source_audit else None,
            'gt_cache': source_audit['gt_cache'] if source_audit else None,
            'source_comparison': source_comparison,
            'source_comparison_counts': dict(Counter(row['classification'] for row in source_comparison)),
            'not_a_file_transfer_or_frontend_freeze': True,
        },
        'dino_checkpoint_recovery': {
            'receipt': file_record(checkpoint_receipt_path),
            'details': checkpoint_receipt,
            'model_inference_or_feature_compatibility_proved': False,
        },
        'formal_parquet_dependency_smoke': {
            'receipt': file_record(dependency_receipt_path),
            'details': dependency_receipt,
            'actual_nas_payload_validation_proved': False,
        },
        'new_environment_dependency_manifest_bytes': dependency_receipt['installed_pyarrow_distribution_bytes']
                                                     if dependency_receipt else 0,
        'new_large_asset_bytes': (ROOT / 'checkpoints/dinov2_vitb14_pretrain.pth').stat().st_size
                                if (ROOT / 'checkpoints/dinov2_vitb14_pretrain.pth').is_file() else 0,
        'legacy_evaluation_limitations': [
            'persistent evaluator accepts contaminated tokens after any same-category history',
            'NEW, wrong EXISTING and wrong KNOWN are collapsed into false assignment',
            'physical local IDs are not namespaced by video in category-track eligibility',
            'predicted evaluator denominator comes from matched join, not all GT opportunities',
            'standard evaluator remaps KNOWN tokens together with anonymous tokens',
            'whole-track ordering/prefix means do not prove truly frame-online replay',
        ],
        'training_permitted_now': False,
        'test_annotation_opened': False, 'external_process_interference': False,
        'large_assets_copied': False,
    }
    atomic_json(args.output, audit)
    print(json.dumps({'output': str(args.output), 'status': audit['status'],
                      'train_known_categories': len(annotations['train']['train_supported_known_ids']),
                      'disk_free_gib': audit['resource_snapshot']['available_disk_bytes'] / 2**30}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
