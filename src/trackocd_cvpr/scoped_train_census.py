"""External Known census only. Video gating precedes target-value decoding.

Byte validation/skipping is not semantic decoding. No whole source object or
annotation body is json.loads'ed. This API is NOT model-facing or an absence
certificate. NonKnown numbers are transient membership filters, not retained.
"""
from collections import Counter, defaultdict
import json
import math
import mmap
import re

from .metadata_only import _space, _end_value, _NUMBER


def object_ranges(data, start, end):
    if data[start] != ord('{'):
        raise ValueError('Object required')
    values, p = {}, _space(data, start + 1)
    while data[p] != ord('}'):
        key_end = _end_value(data, p)
        key = json.loads(data[p:key_end])
        if not isinstance(key, str) or key in values:
            raise ValueError('Duplicate or invalid object key')
        p = _space(data, key_end)
        if data[p] != ord(':'):
            raise ValueError('Object colon required')
        p = _space(data, p + 1)
        value_end = _end_value(data, p)
        values[key] = (p, value_end)
        p = _space(data, value_end)
        if data[p] == ord(','):
            p = _space(data, p + 1)
        elif data[p] != ord('}'):
            raise ValueError('Object separator required')
    if p + 1 != end:
        raise ValueError('Object boundary mismatch')
    return values


def array_ranges(data, start, end):
    if data[start] != ord('['):
        raise ValueError('Array required')
    p = _space(data, start + 1)
    while data[p] != ord(']'):
        value_end = _end_value(data, p)
        yield p, value_end
        p = _space(data, value_end)
        if data[p] == ord(','):
            p = _space(data, p + 1)
        elif data[p] != ord(']'):
            raise ValueError('Array separator required')
    if p + 1 != end:
        raise ValueError('Array boundary mismatch')


def integer_value(data, bounds):
    raw = data[slice(*bounds)]
    if len(raw) > 19 or re.fullmatch(rb'(?:0|[1-9][0-9]*)', raw) is None:
        raise ValueError('Nonnegative bounded integer required before decode')
    return json.loads(raw)


def top_ranges(data):
    start = _space(data, 0)
    end = _end_value(data, start)
    if _space(data, end) != len(data):
        raise ValueError('Trailing JSON input')
    return object_ranges(data, start, end)


def inherited_known_ids(path):
    """Only known_ids numbers; names/Novel IDs/all other values byte-skipped."""
    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        fields = top_ranges(data)
        if 'known_ids' not in fields:
            raise ValueError('Missing inherited Known IDs')
        ids = [integer_value(data, bounds) for bounds in array_ranges(data, *fields['known_ids'])]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Nonempty unique inherited Known IDs required')
    return frozenset(ids)


def validate_video_scope(index, expected_allowed=429, expected_excluded=51):
    if (index.get('schema_version') != 'trackocd.cvpr.private-administrative-video-exclusion.v1'
            or index.get('not_model_facing') is not True or index.get('semantic_target_labels') is not False):
        raise ValueError('Administrative private index required')
    parts = index['partitions']
    if set(parts) != {'prototype', 'representation_fit', 'development', 'policy_train', 'heldout_selection'}:
        raise ValueError('Exact old partitions required')
    owners = {}
    for part, videos in parts.items():
        if not isinstance(videos, list) or len(set(videos)) != len(videos):
            raise ValueError('Unique administrative IDs required')
        for video in videos:
            if type(video) is not int or video < 0 or video in owners:
                raise ValueError('Invalid or overlapping video scope')
            owners[video] = part
    allowed = frozenset(v for v, part in owners.items() if part != 'heldout_selection')
    excluded = frozenset(parts['heldout_selection'])
    if (len(allowed), len(excluded)) != (expected_allowed, expected_excluded):
        raise ValueError('Declared video boundary differs')
    return allowed, excluded


def census(path, allowed_videos, excluded_videos, known_ids, guard=lambda: None):
    if (not allowed_videos or not known_ids or allowed_videos & excluded_videos
            or any(type(v) is not int or v < 0 for v in allowed_videos | excluded_videos | known_ids)):
        raise ValueError('Nonempty valid explicit default-deny scope required')
    counts, tracks, seen_allowed = Counter(), {}, set()
    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        fields = top_ranges(data)
        guard()
        if 'annotations' not in fields:
            raise ValueError('Missing annotation array')
        for start, end in array_ranges(data, *fields['annotations']):
            spans = object_ranges(data, start, end)
            # Targets may occur BEFORE video_id in JSON: spans contain bytes
            # only. No target values have been decoded at this point.
            if 'video_id' not in spans:
                raise ValueError('Administrative video_id required before targets')
            video = integer_value(data, spans['video_id'])
            counts['all_annotation_records_scanned'] += 1
            if counts['all_annotation_records_scanned'] % 256 == 0:
                guard()
            if video not in allowed_videos:
                counts['excluded_heldout_records_byte_skipped' if video in excluded_videos
                       else 'other_unlisted_records_byte_skipped'] += 1
                continue
            seen_allowed.add(video)
            counts['allowed_annotation_records'] += 1
            if 'category_id' not in spans:
                raise ValueError('Category membership filter missing in allowed video')
            category = integer_value(data, spans['category_id'])
            if category not in known_ids:
                counts['allowed_nonKnown_records_not_retained'] += 1
                continue
            if not {'image_id', 'track_id', 'bbox'} <= spans.keys():
                raise ValueError('Known annotation lacks required bookkeeping/box')
            image = integer_value(data, spans['image_id'])
            track = integer_value(data, spans['track_id'])
            raw_box = [data[slice(*bounds)] for bounds in array_ranges(data, *spans['bbox'])]
            if len(raw_box) != 4 or any(len(raw) > 64 or _NUMBER.fullmatch(raw) is None for raw in raw_box):
                raise ValueError('Four scalar numeric Known box coordinates required before decode')
            box = [json.loads(raw) for raw in raw_box]
            if (len(box) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in box)
                    or box[2] <= 0 or box[3] <= 0):
                raise ValueError('Invalid Known xywh box; no GT repair or filtering')
            key = (video, track)
            if key not in tracks:
                tracks[key] = {'video_id': video, 'track_id': track, 'category_id': category,
                               'annotation_count': 0, 'images': set()}
            row = tracks[key]
            if row['category_id'] != category or image in row['images']:
                raise ValueError('Inconsistent category or duplicate track/image annotation')
            row['annotation_count'] += 1
            row['images'].add(image)
            counts['retained_Known_annotation_records'] += 1
        guard()
    private_rows, support, video_categories = [], defaultdict(set), defaultdict(set)
    for key in sorted(tracks):
        row = tracks[key]
        row['distinct_annotated_images'] = len(row.pop('images'))
        private_rows.append(row)
        support[row['category_id']].add(row['video_id'])
        video_categories[row['video_id']].add(row['category_id'])
    public = {'counts': dict(counts), 'allowlisted_videos': len(allowed_videos),
              'allowlisted_videos_with_any_positive_annotations': len(seen_allowed),
              'allowlisted_videos_without_positive_records': len(allowed_videos - seen_allowed),
              'videos_with_retained_Known_tracks': len(video_categories), 'retained_Known_tracks': len(private_rows),
              'inherited_Known_categories': len(known_ids), 'positive_supported_Known_categories': len(support),
              'Known_category_video_support_counts_sorted': sorted(map(len, support.values())),
              'video_positive_Known_category_count_histogram': dict(sorted(Counter(map(len, video_categories.values())).items())),
              'track_observation_count_histogram': dict(sorted(Counter(r['annotation_count'] for r in private_rows).items())),
              'track_cap_applied': False, 'short_tracks_retained': True,
              'positive_annotations_certify_visual_category_absence': False,
              'new_protocol_frozen': False, 'model_executed': False}
    private = {'schema_version': 'trackocd.cvpr.private-known-positive-census.v1',
               'not_model_facing': True, 'tracks': private_rows,
               'video_positive_Known_categories': {str(v): sorted(cs) for v, cs in sorted(video_categories.items())}}
    guard()
    return public, private
