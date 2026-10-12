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
