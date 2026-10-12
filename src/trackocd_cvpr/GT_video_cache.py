"""Uncapped GT-geometric control shards; routing keys never enter model views."""
import json
import mmap
import os
from pathlib import Path

import numpy as np
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.features import prefix_view
from .scoped_train_census import top_ranges, object_ranges, array_ranges, integer_value


def video_plan(path, video, image_plan):
    """Gate video before decoding any observation or opaque routing ID."""
    return selected_video_plans(path, [video], image_plan)[0]


def selected_video_plans(path, videos, image_plan):
    """One projected scan, not one full index scan per small video shard."""
    videos = tuple(videos)
    if not videos or len(set(videos)) != len(videos) or any(type(v) is not int or v < 0 for v in videos):
        raise ValueError('Explicit unique video allowlist')
    grouped = {v: [] for v in videos}
    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for bounds in array_ranges(data, *top_ranges(data)['tracks']):
            fields = object_ranges(data, *bounds)
            video = integer_value(data, fields['video_id'])
            if video not in grouped:
                continue
            track = integer_value(data, fields['opaque_track_id'])
            if track < 0:
                raise ValueError('Nonnegative opaque routing key')
            grouped[video].append({'track_id': track, 'observations': json.loads(data[slice(*fields['observations'])])})
    return [_assemble_video(v, grouped[v], image_plan) for v in videos]


def _assemble_video(video, tracks, image_plan):
    images = sorted((r for r in image_plan['image_metadata'] if r['video_id'] == video),
                    key=lambda r: (r['frame_index'], r['id']))
    if (not images or len({r['id'] for r in images}) != len(images)
            or len({r['frame_index'] for r in images}) != len(images)
            or any(type(r[k]) is not int or r[k] < 0 for r in images for k in ('id', 'frame_index'))):
        raise ValueError('Unique complete metadata frame clock required')
    by_image = {r['id']: r for r in images}
    for track in tracks:
        obs = track['observations']
        if not obs:
            raise ValueError('No empty tracks or filtering')
        last = -1
        for o in obs:
            if (type(o) is not list or len(o) != 6 or any(type(v) is not int for v in o[:2])
                    or any(type(v) not in (int, float) or not np.isfinite(v) for v in o[2:])
                    or min(o[4:]) <= 0 or o[1] <= last or o[0] not in by_image
                    or by_image[o[0]]['frame_index'] != o[1]):
                raise ValueError('Exact finite positive geometry / chronology / image binding')
            last = o[1]

    if not tracks or len({t['track_id'] for t in tracks}) != len(tracks):
        raise ValueError('All unique physical tracks required')
    tracks.sort(key=lambda r: r['track_id'])  # storage only, not frame-policy ordering
    offsets = np.asarray([0] + list(np.cumsum([len(t['observations']) for t in tracks])), np.int64)
    obs = [o for t in tracks for o in t['observations']]
    boxes = np.asarray([[o[2], o[3], o[2] + o[4], o[3] + o[5]] for o in obs], np.float32)
    sizes = np.asarray([[by_image[o[0]]['width'], by_image[o[0]]['height']] * 2 for o in obs], np.float32)
    if np.any(sizes <= 0):
        raise ValueError('Positive current metadata image dimensions')
    arrays = {'visual': np.full((len(obs), 768), np.nan, np.float16),
              'geometry': boxes / sizes, 'quality': np.ones(len(obs), np.float32),
              'frame_index': np.asarray([o[1] for o in obs], np.int64),
              'image_id': np.asarray([o[0] for o in obs], np.int64), 'offsets': offsets,
              'track_id': np.asarray([t['track_id'] for t in tracks], np.int64),
              'clock_frame_index': np.asarray([r['frame_index'] for r in images], np.int64),
              'clock_image_id': np.asarray([r['id'] for r in images], np.int64)}
    return images, boxes, arrays


def validate_arrays(a):
    required = {'visual', 'geometry', 'quality', 'frame_index', 'image_id', 'offsets',
                'track_id', 'clock_frame_index', 'clock_image_id'}
    if set(a) != required:
        raise ValueError('Exact label-free GT control cache schema')
    n = len(a['visual']); tracks = len(a['track_id'])
    shapes = {'visual': (n, 768), 'geometry': (n, 4), 'quality': (n,), 'frame_index': (n,),
              'image_id': (n,), 'offsets': (tracks + 1,), 'track_id': (tracks,),
              'clock_frame_index': (len(a['clock_image_id']),), 'clock_image_id': (len(a['clock_image_id']),)}
    for name, shape in shapes.items():
        dtype = np.float16 if name == 'visual' else np.float32 if name in ('geometry', 'quality') else np.int64
        if a[name].shape != shape or a[name].dtype != dtype or not np.isfinite(a[name]).all():
            raise ValueError('Finite exact dtype/shape ' + name)
    if (n == 0 or tracks == 0 or a['offsets'][0] != 0 or a['offsets'][-1] != n
            or np.any(np.diff(a['offsets']) <= 0) or len(set(a['track_id'].tolist())) != tracks
            or np.any(a['track_id'] < 0) or np.any(a['clock_frame_index'] < 0) or np.any(a['clock_image_id'] < 0)
            or not np.all(a['quality'] == 1) or np.any(a['geometry'][:, 2:] <= a['geometry'][:, :2])
            or np.any(np.diff(a['clock_frame_index']) <= 0)
            or len(set(a['clock_image_id'].tolist())) != len(a['clock_image_id'])):
        raise ValueError('No dropped/empty/duplicate tracks; explicit quality1 and strict frame clock')
    clock = dict(zip(a['clock_image_id'].tolist(), a['clock_frame_index'].tolist()))
    for image, frame in zip(a['image_id'].tolist(), a['frame_index'].tolist()):
        if clock.get(image) != frame:
            raise ValueError('Observation does not bind to complete image/frame clock')
    norms = np.linalg.norm(a['visual'].astype(np.float32), axis=1)
    if np.any(np.abs(norms - 1) > 0.001):
        raise ValueError('Stored common descriptor is not approximately unit')
    for first, end in zip(a['offsets'], a['offsets'][1:]):
        if np.any(np.diff(a['frame_index'][first:end]) <= 0):
            raise ValueError('Every track strictly chronological')


def observed_prefix(a, track_position, observed):
    if type(track_position) is not int or not 0 <= track_position < len(a['track_id']):
        raise ValueError('External track position')
    first, end = (int(x) for x in a['offsets'][track_position:track_position + 2])
    if type(observed) is not int or not 1 <= observed <= end - first:
        raise ValueError('No implicit full-track cap or future request')
    # Materialize only the requested visible part, not a full trajectory.
    visual = np.array(a['visual'][first:first + observed], dtype=np.float32, copy=True)
    visual /= np.linalg.norm(visual, axis=1, keepdims=True)
    return prefix_view(visual, a['geometry'][first:first + observed], a['quality'][first:first + observed],
                       a['frame_index'][first:first + observed], observed)


def check_every_prefix(a):
    validate_arrays(a)
    checks = mutations = 0
    for t, (first, end) in enumerate(zip(a['offsets'], a['offsets'][1:])):
        for observed in range(1, int(end - first) + 1):
            original = observed_prefix(a, t, observed)
            mutated = {k: v.copy() for k, v in a.items()}
            for key in ('visual', 'geometry', 'quality'):
                mutated[key][first + observed:end] = np.nan
            mutated['frame_index'][first + observed:end] = -1
            altered = observed_prefix(mutated, t, observed)
            for key in ('visual', 'boxes_normalized_xyxy', 'quality', 'elapsed_frames'):
                np.testing.assert_array_equal(getattr(original, key), getattr(altered, key))
            checks += 1; mutations += int(first + observed < end)
    return {'every_visible_prefix_checked': checks, 'prefixes_with_actual_future_tail_mutation': mutations,
            'reader_float32_unit_normalization': True, 'not_frame_online_policy_or_trained_gain': True}


def write_atomic_shard(directory, ordinal, arrays, lineage):
    """Payload -> verified reopen -> manifest -> completion; no incomplete shard reuse."""
    validate_arrays(arrays)
    directory = Path(directory)
    stem = 'private_video_' + str(ordinal).zfill(3)
    payload = directory / (stem + '.npz')
    manifest = directory / (stem + '.manifest.json')
    done = directory / (stem + '.done.json')
    if any(p.exists() for p in (payload, manifest, done)):
        raise ValueError('Preserve existing shard / incomplete evidence')
    temporary = directory / (stem + '.partial.' + str(os.getpid()))
    with temporary.open('xb') as handle:
        np.savez(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
    with np.load(temporary, allow_pickle=False) as opened:
        validate_arrays(opened)
    os.replace(temporary, payload)
    record = {'schema_version': 'trackocd.cvpr.GT-video-shard.v1', 'GT_oracle_control_not_Pred': True,
              'labels_or_roles_in_payload': False, 'observations_uncapped': True,
              'payload_completion_not_overall_pilot_precision_qualification': True,
              'tracks': len(arrays['track_id']), 'observations': len(arrays['visual']),
              'metadata_frames': len(arrays['clock_image_id']),
              'empty_GT_frames': len(set(arrays['clock_image_id']) - set(arrays['image_id'])),
              'lineage': lineage, 'payload': {'filename': payload.name, 'bytes': payload.stat().st_size,
                                            'sha256': sha256_file(payload)}}
    atomic_json(manifest, record)
    atomic_json(done, {'complete': True, 'manifest_sha256': sha256_file(manifest),
                       'payload_sha256': record['payload']['sha256']})
    return record


def read_verified_shard(directory, ordinal):
    directory = Path(directory); stem = 'private_video_' + str(ordinal).zfill(3)
    manifest = directory / (stem + '.manifest.json')
    done_path = directory / (stem + '.done.json')
    if manifest.is_symlink() or done_path.is_symlink():
        raise ValueError('No redirected completion files')
    done = json.loads(done_path.read_text())
    record = json.loads(manifest.read_text()); payload = directory / record['payload']['filename']
    if (not done['complete'] or done['manifest_sha256'] != sha256_file(manifest)
            or record['schema_version'] != 'trackocd.cvpr.GT-video-shard.v1'
            or record['GT_oracle_control_not_Pred'] is not True
            or record['labels_or_roles_in_payload'] is not False or record['observations_uncapped'] is not True
            or record['payload']['filename'] != stem + '.npz' or payload.is_symlink()
            or record['payload']['bytes'] != payload.stat().st_size
            or record['payload']['sha256'] != sha256_file(payload)
            or done['payload_sha256'] != record['payload']['sha256']):
        raise ValueError('Incomplete/changed GT shard')
    with np.load(payload, allow_pickle=False) as opened:
        arrays = {k: opened[k] for k in opened.files}
    validate_arrays(arrays)
    if (record['tracks'] != len(arrays['track_id']) or record['observations'] != len(arrays['visual'])
            or record['metadata_frames'] != len(arrays['clock_image_id'])
            or record['empty_GT_frames'] != len(set(arrays['clock_image_id']) - set(arrays['image_id']))):
        raise ValueError('Manifest denominator differs')
    return arrays, record


def validate_reuse_plan(arrays, planned, record, *, video, frozen_sha, physical_sha, images_sha, checkpoint_sha):
    """Exact metadata/lineage binding; source commit may differ, numeric protocol may not."""
    validate_arrays(arrays)
    expected = {'opaque_video_id': video, 'frozen_protocol_sha256': frozen_sha,
                'physical_index_sha256': physical_sha, 'image_plan_sha256': images_sha,
                'checkpoint_sha256': checkpoint_sha}
    if any(record['lineage'].get(k) != v for k, v in expected.items()):
        raise ValueError('Prior shard provenance not this exact frozen GT control')
    for key in set(planned) - {'visual'}:
        np.testing.assert_array_equal(arrays[key], planned[key])
    if arrays['visual'].shape != planned['visual'].shape:
        raise ValueError('Prior shard uncapped visual extent differs')
