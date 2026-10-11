#!/usr/bin/env python3
"""Preregistered small real-cache atomic interface mutations; no training/GT."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import sys
import time

# Set before NumPy/Torch imports; one CPU worker, no GPU model operation.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.atomic_legacy_control import AtomicLegacyControl
from src.trackocd_cvpr.causal_observation_router import CurrentObservation
from src.trackocd_cvpr.diagnostic_feature_cache import DiagnosticFeatureCache

SOURCES = (
    'configs/trackocd_cvpr/cvpr0_atomic_legacy_smoke.json',
    'scripts/trackocd_cvpr/audit_atomic_legacy_control.py',
    'src/trackocd_cvpr/atomic_legacy_control.py',
    'src/trackocd_cvpr/causal_observation_router.py',
    'src/trackocd_cvpr/diagnostic_feature_cache.py',
    'src/trackocd_core/persistent_policy.py',
    'src/trackocd_core/limited_masa_features.py',
    'src/trackocd_core/features.py',
    'src/trackocd_v2/io.py',
    'configs/trackocd_cvpr/autonomous_research.json',
)


def fixed_policy(name):
    if name == 'always_new':
        return lambda cues: 2
    if name == 'existing_if_available_else_new':
        return lambda cues: 1 if cues[16] else 2
    if name == 'always_wait':
        return lambda cues: 3
    raise ValueError('Unregistered fixed diagnostic policy')


def memory_digest(control):
    memory = control.state_snapshot()
    digest = hashlib.sha256(struct.pack('<Q', memory.next_id))
    for token, member in memory.anonymous.items():
        digest.update(token.encode('ascii') + b'\0')
        digest.update(np.asarray(member.vector_sum, dtype='<f4').tobytes())
        digest.update(struct.pack('<Qd', member.count, member.quality_sum))
    return digest.hexdigest()


def probe(reader, frames, cfg, policy_name, mode, guard):
    ordered = sorted(frames)
    cutoff = ordered[len(ordered) // 2 - 1]
    control = AtomicLegacyControl({}, fixed_policy(policy_name),
                                  max_tracks=cfg['max_tracks'],
                                  max_observations=cfg['max_observations'],
                                  max_anonymous=cfg['max_anonymous'])
    digest, prior = hashlib.sha256(), hashlib.sha256()
    actions = Counter()
    queries = groups = ties = 0
    arrays = reader.data

    def step(frame, observations, canonical_ids):
        nonlocal queries, groups, ties
        guard()
        result = control.ingest(frame, observations)
        payload = bytearray(struct.pack('<q', frame))
        for routes, event in zip(result.routing_groups, result.events):
            # Route correspondence is external diagnostic bookkeeping only.
            # It must never choose model group ordering or policy actions.
            payload.extend(json.dumps(sorted(canonical_ids[x] for x in routes)).encode('ascii'))
            payload.extend(bytes.fromhex(event.visible_input_sha256))
            payload.extend(event.cues_float32_bytes)
            payload.extend(event.action.encode('ascii') + b'\0')
            payload.extend((event.token or '').encode('ascii') + b'\0')
            actions[event.action] += 1
        payload.extend(bytes.fromhex(memory_digest(control)))
        digest.update(payload)
        if frame <= cutoff:
            prior.update(payload)
        queries += len(observations)
        groups += len(result.events)
        ties += sum(len(routes) > 1 for routes in result.routing_groups)

    for frame in ordered:
        entries = frames[frame]
        if mode == 'renumber_and_reverse':
            entries = list(reversed(entries))
        observations, canonical_ids = [], {}
        for index, offset in entries:
            identity = (f'opaque_{len(reader.rows) - index}' if mode == 'renumber_and_reverse'
                        else str(reader.rows[index]['physical_track_id']))
            visual = arrays['visual'][offset]
            if mode == 'future_negation' and frame > cutoff:
                visual = -visual
            observations.append(CurrentObservation(identity, visual, arrays['geometry'][offset],
                                                   float(arrays['quality'][offset])))
            canonical_ids[identity] = index
        step(frame, observations, canonical_ids)
    retained_digest = digest.hexdigest()
    if mode == 'future_append':
        last = observations[-1]
        step(ordered[-1] + 1, [
            CurrentObservation(last.routing_id, last.visual, last.box_normalized_xyxy, last.quality),
            CurrentObservation('future_only_external_route', last.visual,
                               last.box_normalized_xyxy, last.quality)],
             {last.routing_id: canonical_ids[last.routing_id], 'future_only_external_route': len(reader.rows)})
    return {'full_output_state_sha256': digest.hexdigest(), 'prior_output_state_sha256': prior.hexdigest(),
            'retained_output_state_sha256_before_append': retained_digest,
            'final_memory_sha256': memory_digest(control), 'queries': queries, 'atomic_groups': groups,
            'exact_tie_groups': ties, 'actions': dict(sorted(actions.items())), 'cutoff_frame': cutoff}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    parser.add_argument('--git-proxy', default='')
    args = parser.parse_args()
    cfg = json.loads((ROOT / SOURCES[0]).read_text())
    safety = json.loads((ROOT / SOURCES[-1]).read_text())
    git = ['git'] + (['-c', f'http.proxy={args.git_proxy}'] if args.git_proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote_rows = subprocess.check_output(
        git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"], cwd=ROOT,
        text=True, timeout=35, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()
    if not remote_rows or remote_rows[0] != args.preregistration_commit:
        raise ValueError('Exact current remote preregistration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{args.preregistration_commit}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered source changed')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve every previous attempt')
    started = time.monotonic()

    def guard():
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']:
            raise RuntimeError('RAM reserve')
        if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg['rss_bytes_ceiling']
                or time.monotonic() - started > cfg['wall_seconds_ceiling']
                or shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes']):
            raise RuntimeError('RSS/wall/free disk bound')
        if out.exists() and sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > cfg['output_bytes_ceiling']:
            raise RuntimeError('Aggregate receipt size bound')
        if sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file()) > safety['incremental_storage_hard_bytes']:
            raise RuntimeError('Incremental project disk bound')

    guard()
    feature_root = ROOT / cfg['feature_directory']
    result_path = feature_root / 'result.json'
    if sha256_file(result_path) != cfg['feature_result_sha256']:
        raise ValueError('Completed feature result changed')
    feature_result = json.loads(result_path.read_text())
    manifest_path = feature_root / feature_result['private_manifest']['filename']
    if sha256_file(manifest_path) != cfg['feature_manifest_sha256'] or feature_result['private_manifest']['sha256'] != cfg['feature_manifest_sha256']:
        raise ValueError('Completed manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if not manifest['complete'] or manifest['records'][0]['video_id'] != cfg['video_id']:
        raise ValueError('Predeclared first record required; no outcome-selected subset')
    record = manifest['records'][0]
    if (record['npz_sha256'], record['tracks'], record['observations']) != (
            cfg['video_npz_sha256'], cfg['expected_video_tracks'], cfg['expected_video_observations']):
        raise ValueError('Registered tiny input changed')
    cache = DiagnosticFeatureCache(ROOT, [record], guard)
    launch = {'preregistration_commit': args.preregistration_commit,
              'source_sha256': {name: sha256_file(ROOT / name) for name in SOURCES},
              'feature_result_sha256': sha256_file(result_path),
              'feature_manifest_sha256': sha256_file(manifest_path),
              'selected_npz_sha256': record['npz_sha256']}
    out.mkdir(parents=True)
    atomic_json(out / 'launch.json', launch)
    try:
        reader = cache.video(cfg['video_id'])
        all_frames = defaultdict(list)
        for index, row in enumerate(reader.rows):
            begin = row['observation_offset']
            for offset in range(begin, begin + row['observation_count']):
                all_frames[int(reader.data['frame_index'][offset])].append((index, offset))
        frames = {frame: all_frames[frame] for frame in sorted(all_frames)[:cfg['maximum_distinct_cached_frames']]}
        if len(frames) < 2:
            raise ValueError('At least two causal frames needed for future mutation')
        cases = []
        for policy in cfg['policies']:
            baseline = probe(reader, frames, cfg, policy, 'original', guard)
            renumbered = probe(reader, frames, cfg, policy, 'renumber_and_reverse', guard)
            negated = probe(reader, frames, cfg, policy, 'future_negation', guard)
            appended = probe(reader, frames, cfg, policy, 'future_append', guard)
            if baseline != renumbered:
                raise AssertionError('Actual routing/action/cue/memory mutation discrepancy')
            if not (baseline['prior_output_state_sha256'] == negated['prior_output_state_sha256'] == appended['prior_output_state_sha256']
                    and baseline['full_output_state_sha256'] != negated['full_output_state_sha256']
                    and baseline['full_output_state_sha256'] != appended['full_output_state_sha256']
                    and baseline['full_output_state_sha256'] == appended['retained_output_state_sha256_before_append']):
                raise AssertionError('Future input changed without required prefix invariance')
            cases.append({'fixed_diagnostic_policy': policy, 'baseline': baseline,
                          'renumber_and_reverse_equal': True,
                          'future_negation_changed_later_not_prior': True,
                          'future_append_retained_all_prior_outputs': True})
        guard()
        result = {'status': 'COMPLETE_TINY_REAL_TRAIN_INPUT_ATOMIC_LEGACY_CONTROL_DIAGNOSTIC',
                  **launch, 'selected_cached_frames': len(frames), 'total_cached_video_frames': len(all_frames),
                  'selected_observations': sum(map(len, frames.values())), 'cases': cases,
                  'GT_or_category_labels_decoded': False, 'Val_or_Test_access': False,
                  'new_feature_extraction': False, 'optimization_steps': 0,
                  'trained_checkpoint_parity': False, 'dense_frontend_acceptance': False,
                  'historical_physical_ID_controller_invariance_remains_FAIL': True,
                  'resources': {'wall_seconds': time.monotonic() - started,
                                'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                'training_GPU_hours': 0}, 'interpretation': cfg['interpretation']}
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'selected_observations': result['selected_observations'],
                          'resources': result['resources']}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error),
                                         'wall_seconds': time.monotonic() - started, 'complete': False})
        raise
    finally:
        cache.close()


if __name__ == '__main__':
    main()
