import json
import pytest
from src.trackocd_cvpr.physical_gt_census import physical_census


def source(tmp_path, extra=None):
    data = {'images': [{'id': 1, 'video_id': 10, 'frame_index': 0},
                       {'id': 2, 'video_id': 10, 'frame_index': 1},
                       {'id': 3, 'video_id': 99, 'frame_index': {'protected': 'do not decode'}}],
            'annotations': [{'category_id': {'not_an_integer': ['Novel must stay skipped']},
                             'image_id': i, 'track_id': 7, 'bbox': [0, 0, 2, 3], 'video_id': 10}
                            for i in (1, 2)],
            'videos': [{'id': 99, 'neg_category_ids': ['protected']}],
            'categories': [{'name': 'not decoded'}]}
    data['annotations'].append({'bbox': 'protected', 'track_id': {'protected': True}, 'video_id': 99})
    if extra:
        extra(data)
    path = tmp_path / 'fixture.json'; path.write_text(json.dumps(data))
    return path


def test_categories_and_protected_fields_byte_skipped_geometry_only(tmp_path):
    public, private = physical_census(source(tmp_path), {10}, {99})
    assert public['physical_tracks'] == 1 and public['counts']['retained_physical_observations'] == 2
    assert public['counts']['old_heldout_records_byte_skipped'] == 1
    assert private['tracks'][0]['observations'] == [[1, 0, 0, 0, 2, 3], [2, 1, 0, 0, 2, 3]]
    assert not private['semantic_category_labels_retained']
    assert not public['category_values_names_Novel_vocabulary_video_GT_arrays_decoded']


def test_no_category_field_required_no_semantic_filter(tmp_path):
    def remove(data):
        for row in data['annotations']:
            row.pop('category_id', None)
    assert physical_census(source(tmp_path, remove), {10}, {99})[0]['physical_tracks'] == 1


def test_invalid_selected_geometry_or_duplicate_refused(tmp_path):
    for mutate in (lambda d: d['annotations'][0].update(bbox=[0, 0, 0, 2]),
                   lambda d: d['annotations'].append(d['annotations'][0].copy())):
        with pytest.raises(ValueError):
            physical_census(source(tmp_path, mutate), {10}, {99})
