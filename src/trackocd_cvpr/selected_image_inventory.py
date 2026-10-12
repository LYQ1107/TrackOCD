"""Selected Train path/stat preflight, no pixel/target/model decoding."""
import mmap
from .scoped_train_census import top_ranges, object_ranges, array_ranges, integer_value
from .metadata_only import _project_object
from .train_pixel_inventory import inventory_paths
from .global_protocol_coverage import audit_global_denominators
from .known_census_protocol import ROLES


def selected_image_metadata(path, selected, guard=lambda: None):
    rows = []
    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        fields = top_ranges(data)
        if 'images' not in fields:
            raise ValueError('Image metadata required')
        for i, (start, end) in enumerate(array_ranges(data, *fields['images'])):
            spans = object_ranges(data, start, end)
            video = integer_value(data, spans['video_id'])
            if video not in selected:
                continue
            row = _project_object(data, start, end, ('id', 'video_id', 'frame_index', 'file_name', 'width', 'height'))
            if set(row) != {'id', 'video_id', 'frame_index', 'file_name', 'width', 'height'}:
                raise ValueError('Exact selected scalar image fields required')
            rows.append(row)
            if i % 256 == 0:
                guard()
        guard()
    return {'images': rows, 'videos': [{'id': v} for v in sorted(selected)]}


def audit_selected_images(path, presence, known_index, wrapper, physical, frame_root, guard=lambda: None):
    audit_global_denominators(presence, known_index, wrapper)
    assignments = wrapper['assignment']['video_ids']
    selected = set(v for videos in assignments.values() for v in videos)
    metadata = selected_image_metadata(path, selected, guard)
    by_image = {}
    for row in metadata['images']:
        if row['id'] in by_image:
            raise ValueError('Duplicate image bookkeeping')
        by_image[row['id']] = (row['video_id'], row['frame_index'])
    referenced, observations = set(), 0
    for track in physical['tracks']:
        if track['video_id'] not in selected:
            raise ValueError('Physical index escapes selected video scope')
        for obs in track['observations']:
            if by_image.get(obs[0]) != (track['video_id'], obs[1]):
                raise ValueError('Physical GT image/frame not in exact selected metadata')
            referenced.add(obs[0]); observations += 1
    role_inventory = {}
    for role in ROLES:
        public, complete = inventory_paths(metadata, set(assignments[role]), frame_root, guard)
        role_inventory[role] = public
    public = {'status': 'SELECTED_TRAIN_IMAGE_PATH_STAT_PREFLIGHT_NOT_DECODED_FEATURES',
              'integer_necessary_conditions': {'class_pool': sum(len(cs) for cs in wrapper['assignment']['classes'].values())},
              'selected_videos': len(selected), 'selected_image_metadata_rows': len(by_image),
              'GT_observations_cross_checked': observations, 'GT_referenced_images': len(referenced),
              'selected_metadata_images_without_GT_reference': len(by_image) - len(referenced),
              'role_pixel_inventory': role_inventory,
              'annotation_category_video_GT_values_decoded': False, 'pixel_bytes_hashed_or_decoded': False,
              'features_loaded_or_extracted': False, 'dense_frame_coverage_proven': False,
              'new_protocol_frozen': False, 'training_GPU_hours': 0}
    return public, {'schema_version': 'trackocd.cvpr.private-selected-images.v1', 'not_model_facing': True,
                    'not_a_frozen_protocol': True, 'image_metadata': metadata['images'],
                    'semantic_labels_or_GT_boxes_in_image_plan': False}
