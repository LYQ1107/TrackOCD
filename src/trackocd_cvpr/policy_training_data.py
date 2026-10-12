"""External policy loss targets and current-only GT-control frame delivery.

Only the registered worker may open real private indices, after source/hash/
authority/cache gates. This module neither runs training nor grants access.
GT localization remains an oracle control, not a predicted-track benchmark.
Labels never enter observations, candidate features or prediction-owned memory.
"""
from collections import Counter
from dataclasses import dataclass
import mmap

import numpy as np

from .causal_observation_router import CurrentObservation
from .scoped_train_census import top_ranges, object_ranges, array_ranges, integer_value


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
