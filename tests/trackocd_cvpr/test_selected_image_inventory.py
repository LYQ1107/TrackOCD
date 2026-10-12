import json
import pytest
from src.trackocd_cvpr.selected_image_inventory import selected_image_metadata


def test_video_gate_precedes_scalar_path_geometry_decode_and_target_arrays_skipped(tmp_path):
    path = tmp_path / 'fixture.json'
    allowed = dict(id=1, video_id=10, frame_index=0, file_name='train/v/a.jpg', width=10, height=20)
    excluded = dict(file_name={'protected': 'do not decode'}, width=['protected'], video_id=99)
    path.write_text(json.dumps({'images': [excluded, allowed], 'annotations': [{'category_id': ['not read']}],
                                'videos': [{'id': 99, 'neg_category_ids': [999]}], 'categories': [{'name': 'not read'}]}))
    meta = selected_image_metadata(path, {10})
    assert meta == {'images': [allowed], 'videos': [{'id': 10}]}


def test_invalid_selected_scalar_not_repaired(tmp_path):
    path = tmp_path / 'fixture.json'
    path.write_text(json.dumps({'images': [dict(id=1, video_id=10, frame_index=0,
        file_name='train/a.jpg', width={'not_scalar': True}, height=20)]}))
    with pytest.raises(ValueError):
        selected_image_metadata(path, {10})
