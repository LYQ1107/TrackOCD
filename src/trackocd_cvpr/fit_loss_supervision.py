"""Fit-only existing Known targets for loss, never feature/forward inputs.

Caller must first verify the frozen source/index SHA and new human permission.
This projection reads neither original annotations, pixels nor other-role
semantic values. Its private mapping is an external loss/prototype binding,
not a dataset category-absence certificate or trained result.
"""
import mmap
from .scoped_train_census import top_ranges, object_ranges, array_ranges, integer_value


def project_fit_loss_targets(assignment_path, known_path, *, expected_classes=5, expected_videos=257, expected_tracks=1158):
    with open(assignment_path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        assignment = object_ranges(data, *top_ranges(data)['assignment'])
        class_roles = object_ranges(data, *assignment['classes'])
        video_roles = object_ranges(data, *assignment['video_ids'])
        classes = sorted(integer_value(data, r) for r in array_ranges(data, *class_roles['fit_known']))
        video_list = [integer_value(data, r) for r in array_ranges(data, *video_roles['fit_known'])]
        videos = set(video_list)
    if (len(classes) != expected_classes or len(set(classes)) != len(classes)
            or len(videos) != expected_videos or len(video_list) != expected_videos):
        raise ValueError('Exact frozen fit class/video counts')
    bindings = {c: index for index, c in enumerate(classes)}
    targets = {}; counts = {'eligible_supervised_tracks': 0, 'eligible_supervised_observations': 0,
                            'fit_other_retained_Known_tracks_without_loss_target': 0}
    with open(known_path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for bounds in array_ranges(data, *top_ranges(data)['tracks']):
            fields = object_ranges(data, *bounds)
            video = integer_value(data, fields['video_id'])
            if video not in videos:
                continue  # other roles /22unassigned / oldheldout body not decoded
            category = integer_value(data, fields['category_id'])
            if category not in bindings:
                counts['fit_other_retained_Known_tracks_without_loss_target'] += 1
                continue  # existing Known outside fit5 is not a BG/Novel target
            track = integer_value(data, fields['track_id'])
            n = integer_value(data, fields['annotation_count'])
            if min(track, n) < 0 or n == 0 or (video, track) in targets:
                raise ValueError('Unique positive fit supervision key')
            targets[(video, track)] = bindings[category]
            counts['eligible_supervised_tracks'] += 1
            counts['eligible_supervised_observations'] += n
    if len(targets) != expected_tracks:
        raise ValueError('Exact fixed fit supervision denominator')
    proof = {**counts, 'classes_for_loss': len(classes), 'other_role_semantic_values_decoded': False,
             'unlabeled_or_other_Known_assigned_background_or_Novel': False,
             'targets_external_to_numeric_forward': True, 'not_cache_or_training_execution': True}
    return bindings, targets, proof


def external_targets_for_physical_tracks(video, track_ids, targets):
    """None is absent semantic supervision, not a new/negative class."""
    return tuple(targets.get((video, int(track))) for track in track_ids)
