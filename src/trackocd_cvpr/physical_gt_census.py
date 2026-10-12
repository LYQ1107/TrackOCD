"""External physical GT projection. Category VALUES are never decoded.

Video gate precedes image/track/box/frame decoding, irrespective of JSON field
order. Byte syntax validation is not target-value decoding. No labels, names,
negative/exhaustiveness video arrays, image paths, pixels, models or training.
"""
from collections import Counter
import json
import math
import mmap
from .scoped_train_census import top_ranges, object_ranges, array_ranges, integer_value
from .metadata_only import _NUMBER


def reconcile_physical_known(presence, known_index, assignment_wrapper, physical_index):
    """External join only; unlabeled physical GT never gets a Novel class."""
    from .global_protocol_coverage import audit_global_denominators
    from .known_census_protocol import ROLES
    audit_global_denominators(presence, known_index, assignment_wrapper)
    if (type(physical_index) is not dict or set(physical_index) != {'schema_version', 'not_model_facing',
            'not_a_frozen_protocol', 'semantic_category_labels_retained', 'tracks'}
            or physical_index['schema_version'] != 'trackocd.cvpr.private-physical-GT.v1'
            or physical_index['not_model_facing'] is not True or physical_index['not_a_frozen_protocol'] is not True
            or physical_index['semantic_category_labels_retained'] is not False):
        raise ValueError('Exact label-free non-frozen physical index required')
    assignment = assignment_wrapper['assignment']
    by_video = {v: role for role, vs in assignment['video_ids'].items() for v in vs}
    classes = {r: set(cs) for r, cs in assignment['classes'].items()}
    pool = set().union(*classes.values())
    known = {(row['video_id'], row['track_id']): row for row in known_index['tracks'] if row['video_id'] in by_video}
    physical, frame_mapping = {}, {}
    counts = {r: {k: {'tracks': 0, 'observations': 0} for k in
        ('fit_Known', 'own_role_pseudo', 'outside_eligible_Known', 'no_retained_Known_label')} for r in ROLES}
    references = []
    for row in physical_index['tracks']:
        if set(row) != {'video_id', 'opaque_track_id', 'observations'}:
            raise ValueError('Exact geometry-only track schema')
        v, t = row['video_id'], row['opaque_track_id']
        if type(v) is not int or type(t) is not int or min(v, t) < 0 or v not in by_video or (v, t) in physical:
            raise ValueError('Unique allowlisted opaque physical key required')
        images, last_frame = set(), -1
        for obs in row['observations']:
            if (type(obs) is not list or len(obs) != 6 or any(type(x) is not int or x < 0 for x in obs[:2])
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in obs[2:])
                    or obs[4] <= 0 or obs[5] <= 0 or obs[0] in images or obs[1] <= last_frame):
                raise ValueError('Finite box, unique image and strict visible chronology')
            if obs[0] in frame_mapping and frame_mapping[obs[0]] != (v, obs[1]):
                raise ValueError('Image/video/frame relationship inconsistent')
            frame_mapping[obs[0]] = (v, obs[1]); images.add(obs[0]); last_frame = obs[1]
        if not images:
            raise ValueError('Empty physical track')
        physical[(v, t)] = len(images)
        role = by_video[v]
        label = known.get((v, t))
        c = label['category_id'] if label is not None else None
        if label is None:
            kind = 'no_retained_Known_label'
        else:
            if label['annotation_count'] != len(images) or label['distinct_annotated_images'] != len(images):
                raise ValueError('Retained Known count differs from same physical track')
            if c in classes[ROLES[0]]:
                kind = 'fit_Known'
            elif c not in pool:
                kind = 'outside_eligible_Known'
            elif c in classes[role]:
                kind = 'own_role_pseudo'
            else:
                raise ValueError('Other-role pseudo-positive conflict')
        counts[role][kind]['tracks'] += 1; counts[role][kind]['observations'] += len(images)
        references.append({'video_id': v, 'opaque_track_id': t, 'role': role,
                           'label_status': kind, 'retained_Known_category_id': c})
    if not set(known) <= physical.keys() or {v for v, _ in physical} != set(by_video):
        raise ValueError('Missing retained Known key or approved video in physical index')
    for role in ROLES:
        counts[role]['all_annotated_physical_GT'] = {
            k: sum(row[k] for row in counts[role].values()) for k in ('tracks', 'observations')}
    public = {'status': 'VERIFIED_ANNOTATED_PHYSICAL_AND_RETAINED_KNOWN_ROLE_DENOMINATORS',
              'integer_necessary_conditions': {'class_pool': len(pool)},
              'role_denominators': counts, 'physical_tracks': len(physical),
              'physical_observations': sum(physical.values()), 'retained_Known_tracks_joined': len(known),
              'additional_geometry_only_tracks': len(physical) - len(known),
              'Known_count_mismatches': 0, 'Known_keys_missing_in_physical_GT': 0,
              'new_nonKnown_semantic_labels_read_or_inferred': False,
              'geometry_only_tracks_declared_background_or_labeled_Novel': False,
              'annotated_physical_GT_counts_not_dense_or_semantic_Novel_truth': True,
              'pixel_feature_availability_proven': False, 'new_protocol_frozen': False,
              'training_GPU_hours': 0}
    return public, {'not_model_facing': True, 'not_a_frozen_protocol': True,
                    'geometry_index_not_duplicated': True, 'track_references': references}


def physical_census(path, allowed_videos, excluded_videos, guard=lambda: None):
    if (not allowed_videos or allowed_videos & excluded_videos
            or any(type(v) is not int or v < 0 for v in allowed_videos | excluded_videos)):
        raise ValueError('Explicit disjoint physical GT scope required')
    counts, images, frames, tracks = Counter(), {}, set(), {}
    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        fields = top_ranges(data)
        if not {'images', 'annotations'} <= fields.keys():
            raise ValueError('Required arrays missing')
        guard()
        for start, end in array_ranges(data, *fields['images']):
            spans = object_ranges(data, start, end)
            video = integer_value(data, spans['video_id'])
            if video not in allowed_videos:
                continue
            image, frame = (integer_value(data, spans[key]) for key in ('id', 'frame_index'))
            if image in images or (video, frame) in frames:
                raise ValueError('Duplicate selected image/frame bookkeeping')
            images[image] = (video, frame); frames.add((video, frame))
        for start, end in array_ranges(data, *fields['annotations']):
            spans = object_ranges(data, start, end)
            video = integer_value(data, spans['video_id'])
            counts['all_annotation_records_byte_scanned'] += 1
            if counts['all_annotation_records_byte_scanned'] % 256 == 0:
                guard()
            if video not in allowed_videos:
                counts['old_heldout_records_byte_skipped' if video in excluded_videos
                       else 'other_unlisted_records_byte_skipped'] += 1
                continue
            image = integer_value(data, spans['image_id'])
            track = integer_value(data, spans['track_id'])
            if image not in images or images[image][0] != video:
                raise ValueError('Selected annotation image/video relationship missing')
            raw_box = [data[slice(*bounds)] for bounds in array_ranges(data, *spans['bbox'])]
            if len(raw_box) != 4 or any(len(raw) > 64 or _NUMBER.fullmatch(raw) is None for raw in raw_box):
                raise ValueError('Four scalar physical xywh coordinates required before decode')
            box = [json.loads(raw) for raw in raw_box]
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in box) or box[2] <= 0 or box[3] <= 0:
                raise ValueError('Invalid physical box, no repair/filtering')
            key = (video, track)
            row = tracks.setdefault(key, {'video_id': video, 'opaque_track_id': track, 'observations': [], '_images': set()})
            if image in row['_images']:
                raise ValueError('Duplicate physical track/image observation')
            row['_images'].add(image)
            row['observations'].append([image, images[image][1], *box])
            counts['retained_physical_observations'] += 1
        guard()
    rows = []
    for key in sorted(tracks):
        row = tracks[key]
        row.pop('_images'); row['observations'].sort(key=lambda obs: obs[1])
        rows.append(row)
    represented = {row['video_id'] for row in rows}
    if represented != allowed_videos:
        raise ValueError('Every allowed video must reconcile with positive physical observations')
    public = {'status': 'CATEGORY_AGNOSTIC_ANNOTATED_PHYSICAL_GT_CENSUS',
              'allowlisted_videos': len(allowed_videos), 'represented_videos': len(represented),
              'selected_image_metadata_rows': len(images), 'physical_tracks': len(rows),
              'counts': dict(counts), 'single_observation_tracks': sum(len(row['observations']) == 1 for row in rows),
              'category_values_names_Novel_vocabulary_video_GT_arrays_decoded': False,
              'image_paths_pixels_features_models_read': False, 'dense_frame_coverage_proven': False,
              'new_protocol_frozen': False, 'training_GPU_hours': 0}
    private = {'schema_version': 'trackocd.cvpr.private-physical-GT.v1', 'not_model_facing': True,
               'semantic_category_labels_retained': False, 'tracks': rows}
    guard()
    return public, private
