#!/usr/bin/env python3
"""Fixed old semantics: all predicted IDs vs selected/capped Train reference."""
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
SOURCES = ('configs/trackocd_cvpr/cvpr0_matched_train_semantics.json',
    'scripts/trackocd_cvpr/compare_matched_train_semantics.py',
    'src/trackocd_cvpr/diagnostic_feature_cache.py', 'src/trackocd_cvpr/truncated_reference_join.py',
    'src/trackocd_core/train_first_replay.py', 'src/trackocd_core/train_first_experiment.py',
    'src/trackocd_core/limited_masa_features.py', 'src/trackocd_core/sealed_parquet.py',
    'src/trackocd_core/evaluation/contracts.py', 'src/trackocd_core/evaluation/standard.py',
    'src/trackocd_core/evaluation/persistent.py', 'configs/trackocd_cvpr/autonomous_research.json')


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
        raise ValueError('Exact remote matched-reference registration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{remote}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered matched-reference source changed')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve all actual attempts')
    import torch
    import pyarrow.parquet as pq
    from src.trackocd_core.train_first_cache import TrainFirstCache
    from src.trackocd_core.train_first_experiment import frozen_model, evidence_bank, prototypes
    from src.trackocd_core.train_first_replay import replay
    from src.trackocd_core.persistent_policy import DecisionMLP
    from src.trackocd_core.sealed_parquet import write_sealed, read_sealed
    from src.trackocd_core.evaluation import Target, TrackKey, join_evaluation, evaluate_standard, evaluate_persistent
    from src.trackocd_core.evaluation.persistent import fixed_cross_video_targets
    from src.trackocd_cvpr.diagnostic_feature_cache import DiagnosticFeatureCache
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
        if out.exists() and sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) > cfg['private_ledger_bytes_ceiling']:
            raise RuntimeError('Bounded diagnostic output')
        if sum(f.stat().st_size for f in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if f.is_file()) > safety['incremental_storage_hard_bytes']:
            raise RuntimeError('CVPR retained incremental storage ceiling')

    guard()
    historical = ROOT / 'outputs/trackocd_core/core_training/train_first_v1'
    freeze_path = historical / 'model_freeze.json'
    if sha256_file(freeze_path) != cfg['model_freeze_sha256']:
        raise ValueError('Protected model registry changed')
    for name, digest in json.loads(freeze_path.read_text())['protected_sha256'].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError('Protected input/source/model changed')
    feature_root = ROOT / cfg['feature_directory']
    feature_result_path = feature_root / 'result.json'
    feature_result = json.loads(feature_result_path.read_text())
    if (feature_result['status'] != 'COMPLETE_BOUNDED_TRAIN_ALL_ID_FEATURES_NOT_MAIN_QUALIFICATION'
            or feature_result['preregistration_commit'] != cfg['feature_preregistration_commit']
            or feature_result['config_sha256'] != cfg['feature_config_sha256']):
        raise ValueError('Only actually complete registered all-ID feature diagnostic')
    for name, digest in feature_result['source_sha256'].items():
        registered = subprocess.check_output(['git', 'show', f"{cfg['feature_preregistration_commit']}:{name}"], cwd=ROOT)
        import hashlib
        if hashlib.sha256(registered).hexdigest() != digest:
            raise ValueError('Completed feature receipt differs from registered archived bytes')
    feature_manifest_path = feature_root / feature_result['private_manifest']['filename']
    if sha256_file(feature_manifest_path) != feature_result['private_manifest']['sha256']:
        raise ValueError('Completed feature manifest changed')
    manifest = json.loads(feature_manifest_path.read_text())
    if not manifest['complete']:
        raise ValueError('No partial favourable video selection')
    predicted = DiagnosticFeatureCache(ROOT, manifest['records'], guard)
    order = list(predicted.paths)
    if (len(order), len(predicted.rows), predicted.observations) != (
            cfg['expected_videos'], cfg['expected_predicted_tracks'], cfg['expected_observations']):
        raise ValueError('Full registered feature universe required')
    cache_root = ROOT / 'outputs/trackocd_core/features/train_first_v1'
    if sha256_file(cache_root / 'manifest.json') != cfg['feature_manifest_sha256']:
        raise ValueError('Protected capped GT reference changed')
    reference = TrainFirstCache(cache_root)
    source = pq.read_table(cache_root / 'train_labels.parquet',
                          filters=[('partition', 'in', cfg['allowed_label_partitions'])]).to_pylist()
    routes = {r['key']: r for r in reference.rows}
    labels = [{**r, 'video_id': routes[r['key']]['video_id']} for r in source]
    by_key = {r['key']: r for r in labels}
    known = sorted({r['category_id'] for r in labels if r['partition'] == 'representation_fit'})
    gt_rows = [r for r in reference.rows if r['key'] in by_key and by_key[r['key']]['partition'] == 'development']
    roles = Counter(by_key[r['key']]['simulation_role'] for r in gt_rows)
    if (len(gt_rows), roles['known'], roles['pseudo_novel']) != (
            cfg['expected_GT_reference_tracks'], cfg['expected_Known_references'], cfg['expected_pseudo_Novel_references']):
        raise ValueError('Full frozen development reference roles/denominators differ')
    if set(r['video_id'] for r in gt_rows) != set(order):
        raise ValueError('GT and predicted input video universe must match')
    receipt = json.loads((historical / 'policy/training_receipt.json').read_text())
    fits = {(f['representation'], f['seed'], f['model']): f for f in receipt['fits']}
    out.mkdir(parents=True)
    ledgers = out / 'private_ledgers'
    ledgers.mkdir()
    launch = {'preregistration_commit': remote, 'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCES},
        'feature_result_sha256': sha256_file(feature_result_path),
        'feature_manifest_sha256': sha256_file(feature_manifest_path), 'video_order': order,
        'no_GPU_or_optimizer': True, 'feature_hash_measured_after_real_completion_not_guessed': True}
    atomic_json(out / 'launch.json', launch)
    cases = []
    # Seal ALL predictions and input counterfactuals before reference matching.
    for representation in cfg['representations']:
        for seed in cfg['seeds']:
            guard()
            rep, lineage = frozen_model(ROOT, representation, seed)
            proto = prototypes(reference, labels, known, rep)
            fit = fits[(representation, seed, cfg['policy'])]
            cp = fit['selected_checkpoint']['checkpoint']
            if sha256_file(ROOT / cp['path']) != cp['sha256']:
                raise ValueError('Frozen policy checkpoint differs')
            model = DecisionMLP().eval().requires_grad_(False)
            model.load_state_dict(torch.load(ROOT / cp['path'], map_location='cpu', weights_only=True)['state_dict'])
            bias = fit['operating_points']['1.0']['wait_bias']
            pred_bank, gt_bank = evidence_bank(predicted, predicted.rows, rep), evidence_bank(reference, gt_rows, rep)
            for prefix in cfg['prefixes']:
                guard()
                identity = f'{representation}_{seed}_p{prefix}'
                case = {'representation': representation, 'seed': seed, 'prefix': prefix, 'wait_bias': bias, 'lineage': lineage}
                for tag, rows, bank in (('predicted', predicted.rows, pred_bank), ('GT_controlled', gt_rows, gt_bank)):
                    sealed, runtime = replay(rows, order, prefix, proto, {}, bank[prefix], decision_model=model, wait_bias=bias)
                    path = ledgers / f'{identity}_{tag}.parquet'
                    write_sealed(path, sealed)
                    if read_sealed(path, order, known).events != sealed.events or len(sealed.events) != len(rows):
                        raise AssertionError('Real seal roundtrip/full universe required')
                    case[tag] = {'runtime': runtime, 'action_counts': dict(Counter(e.kind for e in sealed.events)),
                        'ledger': {'filename': path.name, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}}
                cases.append(case)
                atomic_json(out / 'progress.json', {'status': 'RUNNING_FROZEN_MATCHED_REFERENCE_SEALING',
                    'completed_cases': len(cases), 'wall_seconds': time.monotonic() - started})
            del pred_bank, gt_bank
    predicted.close()
    if len(cases) != cfg['expected_cases']:
        raise ValueError('Incomplete matched-reference case universe')
    matched, audits = {}, []
    for video in order:
        guard()
        gt_geometry, pred_geometry = defaultdict(list), defaultdict(list)
        for row in gt_rows:
            if row['video_id'] == video:
                view = reference.get_prefix(row['key'], 16)
                for frame, box in zip(row['frame_ids'], view.boxes_normalized_xyxy):
                    gt_geometry[int(frame)].append((row['key'], box))
        reader = predicted.video(video)
        for row in reader.rows:
            i, n = row['observation_offset'], row['observation_count']
            for frame, box in zip(reader.data['frame_index'][i:i+n], reader.data['geometry'][i:i+n]):
                pred_geometry[int(frame)].append((row['key'], box))
        mc = cfg['matching']
        pairs, audit = truncated_join(gt_geometry, pred_geometry, mc['IoU_threshold'],
            mc['minimum_shared_observations'], mc['minimum_fraction_of_capped_GT'])
        matched.update(pairs)
        audits.append({'video_id': video, **audit})
    predicted.close()
    gt_keys = {r['key']: TrackKey(r['video_id'], str(r['physical_track_id'])) for r in gt_rows}
    targets = [Target(gt_keys[r['key']], int(by_key[r['key']]['category_id']),
        'known' if by_key[r['key']]['simulation_role'] == 'known' else 'novel') for r in gt_rows]
    matches = {TrackKey(r['video_id'], str(r['physical_track_id'])):
        gt_keys[matched[r['key']]] if r['key'] in matched else None for r in predicted.rows}
    expected_ct = len(fixed_cross_video_targets(targets, order))
    for case in cases:
        guard()
        for tag in ('predicted', 'GT_controlled'):
            sealed = read_sealed(ledgers / case[tag]['ledger']['filename'], order, known)
            join = join_evaluation(sealed, targets, matches if tag == 'predicted' else {t.key: t.key for t in targets})
            standard, persistent = evaluate_standard(join), evaluate_persistent(join)
            standard.pop('global_anonymous_hungarian_mapping_evaluator_only')
            if (standard['old_denominator'], standard['new_denominator'], persistent['commit_ct_denominator']) != (
                    cfg['expected_Known_references'], cfg['expected_pseudo_Novel_references'], expected_ct):
                raise AssertionError('Full fixed cached GT denominators must not shrink')
            case[tag]['evaluation'] = {'standard': standard, 'persistent': persistent}
    guard()
    result = {'status': 'COMPLETE_FROZEN_MATCHED_TRAIN_CAPPED_REFERENCE_DIAGNOSTIC_NOT_MAIN_ACCEPTANCE',
        **launch, 'config_sha256': sha256_file(ROOT / SOURCES[0]), 'cases': cases,
        'matching': {'per_video': audits, 'canonical_reference_targets_matched': len(matched),
            'unknown_unmatched_predicted_tracks': len(predicted.rows) - len(matched)},
        'selected_truncated_reference': {'targets': len(targets), 'Known': roles['known'], 'pseudo_Novel': roles['pseudo_novel'],
            'cross_video_reuse_opportunities': expected_ct},
        'all_predicted_IDs_sealed_before_reference_matching': True, 'GT_in_predicted_inference': False,
        'all_missing_GT_references_in_denominator': True, 'old_numeric_ID_routing_not_repaired': True,
        'GT_counterfactual_changes_stream_and_routing_not_unique_causal_proof': True,
        'Val_or_Test_access': False, 'final_holdout_labels_read': False, 'optimization_steps': 0,
        'resources': {'wall_seconds': time.monotonic() - started,
            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'training_GPU_hours': 0},
        'interpretation': cfg['interpretation']}
    atomic_json(out / 'result.json', result)
    print(json.dumps({'status': result['status'], 'cases': len(cases),
        'matched_references': len(matched), 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    main()
