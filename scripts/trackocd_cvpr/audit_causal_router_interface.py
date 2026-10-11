#!/usr/bin/env python3
"""Registered current-input scheduling mutations on retained Train features."""
import argparse
from collections import defaultdict
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
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.causal_observation_router import CurrentObservation, CurrentObservationRouter
from src.trackocd_cvpr.diagnostic_feature_cache import DiagnosticFeatureCache

SOURCES = ('configs/trackocd_cvpr/cvpr0_causal_router_interface.json',
    'scripts/trackocd_cvpr/audit_causal_router_interface.py',
    'src/trackocd_cvpr/causal_observation_router.py', 'src/trackocd_cvpr/diagnostic_feature_cache.py',
    'src/trackocd_core/limited_masa_features.py', 'src/trackocd_core/features.py',
    'configs/trackocd_cvpr/autonomous_research.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preregistration-commit', required=True)
    parser.add_argument('--git-proxy', default='')
    args = parser.parse_args()
    cfg = json.loads((ROOT / SOURCES[0]).read_text())
    safety = json.loads((ROOT / SOURCES[-1]).read_text())
    git = ['git'] + (['-c', f'http.proxy={args.git_proxy}'] if args.git_proxy else []) + ['-c', 'http.version=HTTP/1.1']
    remote = subprocess.check_output(git + ['ls-remote', 'origin', f"refs/heads/{safety['branch']}"],
        text=True, timeout=35, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}).split()[0]
    if remote != args.preregistration_commit:
        raise ValueError('Exact current remote preregistration required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', f'{remote}:{name}'], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError('Registered router source changed')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Preserve all attempts')
    started = time.monotonic()

    def guard():
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable'] < safety['system_ram_reserve_fraction'] * mem['MemTotal']:
            raise RuntimeError('RAM reserve')
        if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg['rss_bytes_ceiling']
                or time.monotonic() - started > cfg['wall_seconds_ceiling']
                or shutil.disk_usage(ROOT).free < safety['disk_free_reserve_bytes']):
            raise RuntimeError('RSS/wall/disk bound')
        if out.exists() and sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > cfg['output_bytes_ceiling']:
            raise RuntimeError('Small receipt bound')
        if sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file()) > safety['incremental_storage_hard_bytes']:
            raise RuntimeError('Project incremental hard bound')

    guard()
    feature_root = ROOT / cfg['feature_directory']
    feature_result_path = feature_root / 'result.json'
    if sha256_file(feature_result_path) != cfg['feature_result_sha256']:
        raise ValueError('Measured completed feature receipt differs')
    feature_result = json.loads(feature_result_path.read_text())
    manifest_path = feature_root / feature_result['private_manifest']['filename']
    if not sha256_file(manifest_path) == cfg['feature_manifest_sha256'] == feature_result['private_manifest']['sha256']:
        raise ValueError('All-ID completed feature manifest differs')
    manifest = json.loads(manifest_path.read_text())
    if not manifest['complete']:
        raise ValueError('No favourable partial video selection')
    cache = DiagnosticFeatureCache(ROOT, manifest['records'], guard)
    if (len(cache.paths), len(cache.rows), cache.observations) != (cfg['expected_videos'], cfg['expected_tracks'], cfg['expected_observations']):
        raise ValueError('Whole registered limited feature universe required')
    out.mkdir(parents=True)
    launch = {'preregistration_commit': remote, 'source_sha256': {p: sha256_file(ROOT / p) for p in SOURCES},
              'feature_result_sha256': sha256_file(feature_result_path), 'feature_manifest_sha256': sha256_file(manifest_path)}
    atomic_json(out / 'launch.json', launch)
    cases, future_checks = [], None
    try:
        for video_number, video in enumerate(cache.paths):
            guard()
            reader = cache.video(video)
            arrays, frames = reader.data, defaultdict(list)
            for index, row in enumerate(reader.rows):
                begin = row['observation_offset']
                for offset in range(begin, begin + row['observation_count']):
                    frames[int(arrays['frame_index'][offset])].append((index, offset))
            ordered_frames = sorted(frames)
            cutoff = ordered_frames[len(ordered_frames) // 2]

            def probe(mode):
                router = CurrentObservationRouter(max_tracks=cfg['max_observed_tracks_per_video'],
                                                   max_observations=cfg['max_observations_per_video'])
                digest, prior = hashlib.sha256(), hashlib.sha256()
                queries = groups = ties = 0
                for frame in ordered_frames:
                    entries = frames[frame] if mode != 'renumber' else list(reversed(frames[frame]))
                    observations = []
                    for index, offset in entries:
                        identity = str(reader.rows[index]['physical_track_id']) if mode != 'renumber' else f'opaque_{len(reader.rows)-index}'
                        visual = arrays['visual'][offset]
                        if mode == 'future_negation' and frame > cutoff:
                            visual = -visual
                        observations.append(CurrentObservation(identity, visual, arrays['geometry'][offset], float(arrays['quality'][offset])))
                    batch = router.ingest(frame, observations)
                    payload = struct.pack('<q', frame)
                    for view, routes, signature in zip(batch.model_views, batch.routing_groups, batch.visible_input_sha256):
                        payload += bytes.fromhex(signature) + struct.pack('<Q', len(routes))
                        payload += np.asarray(view.weighted_mean(), dtype='<f4').tobytes()
                    digest.update(payload)
                    if frame <= cutoff:
                        prior.update(payload)
                    queries += len(observations)
                    groups += len(batch.model_views)
                    ties += sum(len(g) > 1 for g in batch.routing_groups)
                if mode == 'future_append':
                    last = observations[-1]
                    extra = router.ingest(ordered_frames[-1] + 1, [
                        CurrentObservation(last.routing_id, -last.visual, last.box_normalized_xyxy, last.quality),
                        CurrentObservation('new_future_only_track', last.visual, last.box_normalized_xyxy, last.quality)])
                    digest.update(struct.pack('<q', extra.frame_index))
                    for signature in extra.visible_input_sha256:
                        digest.update(bytes.fromhex(signature))
                return {'digest': digest.hexdigest(), 'prior_digest': prior.hexdigest(),
                        'queries': queries, 'atomic_groups': groups, 'exact_tie_groups': ties}

            original, changed = probe('original'), probe('renumber')
            if original != changed:
                raise AssertionError('Actual opaque-ID/row-permutation input or probe-output change')
            cases.append({'video_number': video_number, 'tracks': len(reader.rows),
                          **original, 'opaque_ID_and_same_frame_permutation_equal': True})
            if video_number == 0:
                negated, appended = probe('future_negation'), probe('future_append')
                if not (negated['prior_digest'] == appended['prior_digest'] == original['prior_digest']
                        and negated['digest'] != original['digest'] and appended['digest'] != original['digest']):
                    raise AssertionError('Actual later input changed without unchanged prior prefix')
                future_checks = {'future_negation_changed_later_not_prior': True,
                                 'future_track_and_observation_append_changed_later_not_prior': True,
                                 'actual_cutoff_frame': cutoff, 'prior_digest': original['prior_digest']}
            atomic_json(out / 'progress.json', {'videos_completed': len(cases), 'elapsed_seconds': time.monotonic() - started})
        guard()
        result = {'status': 'COMPLETE_CURRENT_INPUT_ROUTER_MUTATIONS_NOT_CONTROLLER_OR_DENSE_FRONTEND_ACCEPTANCE',
            **launch, 'cases': cases, 'future_mutations': future_checks,
            'GT_or_labels_decoded': False, 'Val_or_Test_access': False, 'optimization_steps': 0,
            'old_physical_ID_controller_invariance_remains_FAIL': True,
            'resources': {'wall_seconds': time.monotonic() - started,
                          'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'training_GPU_hours': 0},
            'interpretation': cfg['interpretation']}
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'videos': len(cases), 'resources': result['resources']}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error),
                                           'elapsed_seconds': time.monotonic() - started, 'complete': False})
        raise
    finally:
        cache.close()


if __name__ == '__main__':
    main()
