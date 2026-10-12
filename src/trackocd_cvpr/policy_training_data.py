"""External policy loss targets and current-only GT-control frame delivery.

Only the registered worker may open real private indices, after source/hash/
authority/cache gates. This module neither runs training nor grants access.
GT localization remains an oracle control, not a predicted-track benchmark.
Labels never enter observations, candidate features or prediction-owned memory.
"""
from collections import Counter
from dataclasses import dataclass
import json
import mmap
from pathlib import Path

import numpy as np

from .causal_observation_router import CurrentObservation
from .scoped_train_census import top_ranges, object_ranges, array_ranges, integer_value
from .GT_video_cache import read_verified_shard
from src.trackocd_v2.io import sha256_file


@dataclass(frozen=True, slots=True)
class PolicyLossTargets:
    # All fields here are PRIVATE external loss bookkeeping, not model inputs.
    fit_bindings: dict
    pseudo_bindings: dict
    videos: frozenset
    targets: dict


def project_policy_loss_targets(assignment_path, known_path, *, fit_classes=5,
                                pseudo_classes=2, videos=10, known_tracks=34,
                                pseudo_tracks=10):
    """Byte-project fit5/policy2 classes and policy10 videos only.

    The entire non-policy track body is skipped before category/track decode.
    No development/final class or video values, names or nonKnown GT opened.
    """
    with open(assignment_path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        assignment = object_ranges(data, *top_ranges(data)['assignment'])
        classes = object_ranges(data, *assignment['classes'])
        video_roles = object_ranges(data, *assignment['video_ids'])
        fit = [integer_value(data, b) for b in array_ranges(data, *classes['fit_known'])]
        pseudo = [integer_value(data, b) for b in array_ranges(data, *classes['policy_train_pseudo'])]
        selected = [integer_value(data, b) for b in array_ranges(data, *video_roles['policy_train_pseudo'])]
    if (len(fit) != fit_classes or len(pseudo) != pseudo_classes or len(selected) != videos
            or len(set(fit + pseudo)) != fit_classes + pseudo_classes
            or len(set(selected)) != videos or min(fit + pseudo + selected) < 0):
        raise ValueError('Exact disjoint frozen policy classes/videos')
    fit_bindings = dict(zip(fit, range(fit_classes)))
    pseudo_bindings = dict(zip(pseudo, range(pseudo_classes)))
    selected = frozenset(selected)
    targets = {}; counts = Counter()
    with open(known_path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for bounds in array_ranges(data, *top_ranges(data)['tracks']):
            fields = object_ranges(data, *bounds)
            video = integer_value(data, fields['video_id'])
            if video not in selected:
                continue  # no other video's semantic/track/count value decode
            category = integer_value(data, fields['category_id'])
            track = integer_value(data, fields['track_id']); key = (video, track)
            if track < 0 or key in targets:
                raise ValueError('Unique nonnegative physical policy loss key')
            if category in fit_bindings:
                target = ('Known', fit_bindings[category])
            elif category in pseudo_bindings:
                target = ('policy_pseudo', pseudo_bindings[category])
            else:
                target = ('outside_eligible_Known', None)
            targets[key] = target; counts[target[0]] += 1
    if counts['Known'] != known_tracks or counts['policy_pseudo'] != pseudo_tracks:
        raise ValueError('Full frozen policy Known/pseudo target denominators')
    return PolicyLossTargets(fit_bindings, pseudo_bindings, selected, targets), {
        'existing_Known_target_counts': dict(counts),
        'development_final_or_other_video_semantic_values_decoded': False,
        'new_nonKnown_semantic_values_or_vocabulary_decoded': False,
        'external_loss_only_not_observation_or_model_features': True,
    }


def reconcile_policy_targets(physical_keys, projection, *, physical_tracks=54,
                             masked_tracks=10):
    keys = tuple(physical_keys)
    if (len(keys) != physical_tracks or len(set(keys)) != physical_tracks
            or {v for v, _ in keys} != set(projection.videos)
            or not set(projection.targets) <= set(keys)):
        raise ValueError('Full policy physical universe and all target joins required')
    targets = [projection.targets.get(key, ('no_retained_semantic_truth', None)) for key in keys]
    counts = Counter(kind for kind, _ in targets)
    masked = sum(target is None for _, target in targets)
    if masked != masked_tracks:
        raise ValueError('Masked tracks retained, never inferred Novel/background')
    return targets, {'physical_tracks': physical_tracks, 'masked_loss_tracks': masked,
                     'physical_label_strata': dict(counts),
                     'masked_tracks_dropped_or_declared_background': False}


def load_completed_policy_videos(root, cfg, guard):
    """Private policy geometry/targets only, following caller registration.

    Source/authority registration is the worker's responsibility. Cache success,
    protocol, aggregate denominators, completion hashes and routing are required
    here before data reach the numeric frame adapter. No dev NPZ body opened.
    """
    root = Path(root); directory = root / cfg['policy_dev_cache_directory']
    result_path = directory / 'result.json'
    if result_path.is_symlink() or sha256_file(result_path) != cfg['policy_dev_result_sha256']:
        raise ValueError('Immutable qualified policy/dev aggregate receipt')
    result = json.loads(result_path.read_text())
    terminal = json.loads(directory.with_name(directory.name + '_supervisor_attempt_1.json').read_text())
    if (result['status'] != 'VERIFIED_POLICY_DEV_UNCAPPED_GT_CONTROL_CACHE_NOT_FINAL_OR_PRED_OR_TRAINED'
            or result['live_remote_registration_proof']['sha'] != cfg['policy_dev_execution_commit']
            or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None or result['independent_final_or_protected_pixels_read']
            or result['development_or_final_semantic_values_decoded']
            or not result['frozen_parameters_unchanged']
            or result['frozen_parameter_sha256'] != '42c4c180dbaa825ee5bfecc6e6bf392af6439a1d2aa97f05095ac09b5e7097a9'
            or result['optimizer_or_formal_training_started']):
        raise ValueError('Real successful frozen-cache execution, no protected data')
    expected = {'videos': 10, 'tracks': 54, 'observations': 1268, 'metadata_frames': 374, 'empty_GT_frames': 1}
    if any(result['role_totals']['policy_train_pseudo'][key] != value for key, value in expected.items()):
        raise ValueError('Full frozen policy denominators')
    frozen_path = root / cfg['frozen_protocol']
    frozen = json.loads(frozen_path.read_text())
    if frozen['status'] != 'FROZEN_DATA_ROLES_AND_GT_CONTROL_CACHE_IO_NOT_FINAL_METHOD':
        raise ValueError('Frozen authorized data protocol required')
    for field, key in (('private_assignment', 'private_global_assignment'),
                       ('private_Known_support', 'private_Known_support')):
        path = root / cfg[field]
        if path.is_symlink() or sha256_file(path) != frozen['input_sha256'][key]:
            raise ValueError('Immutable existing Known targets/assignment')
    projection, proof = project_policy_loss_targets(root / cfg['private_assignment'], root / cfg['private_Known_support'])
    index_path = directory / 'private_shard_index.json'
    if index_path.is_symlink(): raise ValueError('No redirected private routing')
    index = json.loads(index_path.read_text())  # administrative video/ordinal routing only
    if (len(index) != 82 or len({r['video_id'] for r in index}) != 82
            or sorted(r['ordinal'] for r in index) != list(range(82))):
        raise ValueError('Complete qualified input routing, no duplicate aliases')
    selected = [r for r in index if r['video_id'] in projection.videos]
    if len(selected) != 10 or {r['video_id'] for r in selected} != set(projection.videos):
        raise ValueError('Every authorized policy video, no successful subset')
    public = {r['ordinal']: r for r in result['shards']}
    if len(public) != 82: raise ValueError('All registered public shard receipts')
    videos = []; physical_keys = []; observations = frames = empty = 0
    for row in sorted(selected, key=lambda r: r['ordinal']):
        guard()
        if row['source_directory'] != cfg['policy_dev_cache_directory']:
            raise ValueError('No alternate/redirected payload source')
        arrays, record = read_verified_shard(directory, row['ordinal'])
        lineage = record['lineage']
        required = {'opaque_video_id': row['video_id'], 'frozen_protocol_sha256': sha256_file(frozen_path),
                    'physical_index_sha256': frozen['input_sha256']['private_physical_GT'],
                    'image_plan_sha256': frozen['input_sha256']['private_image_plan'],
                    'checkpoint_sha256': frozen['GT_control_feature_IO']['checkpoint_sha256']}
        if (any(lineage.get(k) != v for k, v in required.items())
                or record['payload']['sha256'] != row['payload_sha256']
                or record['payload']['sha256'] != public[row['ordinal']]['private_payload_sha256']):
            raise ValueError('Exact frozen oracle-control lineage and private/public payload binding')
        videos.append((row['video_id'], arrays))
        physical_keys.extend((row['video_id'], int(t)) for t in arrays['track_id'])
        observations += len(arrays['visual']); frames += len(arrays['clock_frame_index'])
        empty += record['empty_GT_frames']
    _, joined = reconcile_policy_targets(physical_keys, projection)
    if (observations, frames, empty) != (1268, 374, 1):
        raise ValueError('Full physical observations and empty-frame clock retained')
    return videos, projection, {**proof, **joined, 'videos': 10, 'observations': observations,
        'metadata_frames': frames, 'empty_GT_frames': empty,
        'development_final_or_other_video_payload_bodies_opened': False,
        'GT_localization_oracle_control_not_qualified_Pred': True}


def policy_current_frames(arrays, *, routing_bindings=None):
    """Deliver ALL metadata frames, including empty ones, and current rows only.

    Called on an already verified label-free video shard. Offline frame indices
    are a delivery plan, not a model's future-length feature. The callback sees
    CurrentObservation only; total track length and loss targets are absent.
    IDs route state externally and never determine numeric scoring order.
    """
    identities = [int(i) for i in arrays['track_id']]
    routes = {i: str(i) for i in identities} if routing_bindings is None else dict(routing_bindings)
    if (set(routes) != set(identities) or any(type(r) is not str or not r for r in routes.values())
            or len(set(routes.values())) != len(identities)):
        raise ValueError('Bijective external physical routing, not model features')
    positions = {}
    for identity, first, end in zip(identities, arrays['offsets'], arrays['offsets'][1:]):
        for position in range(int(first), int(end)):
            positions.setdefault(int(arrays['frame_index'][position]), []).append((identity, position))
    for frame in arrays['clock_frame_index']:
        observations = []
        for identity, position in positions.get(int(frame), ()):
            visual = np.array(arrays['visual'][position], dtype=np.float32, copy=True)
            norm = np.linalg.norm(visual)
            if not np.isfinite(visual).all() or not np.isfinite(norm) or norm <= 1e-12:
                raise ValueError('Finite current common768 descriptor')
            visual /= norm
            observations.append(CurrentObservation(routes[identity], visual,
                np.array(arrays['geometry'][position], dtype=np.float32, copy=True),
                float(arrays['quality'][position])))
        yield int(frame), tuple(observations)
