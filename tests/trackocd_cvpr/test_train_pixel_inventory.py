"""Synthetic path/stat tests; no real dataset pixels are opened."""
import pytest
from src.trackocd_cvpr.train_pixel_inventory import inventory_paths


def metadata():
    return {'videos': [{'id': 1}, {'id': 2}],
            'images': [{'id': 10, 'video_id': 1, 'file_name': 'train/a.jpg'},
                       {'id': 20, 'video_id': 2, 'file_name': 'train/b.jpg'},
                       {'id': 21, 'video_id': 2, 'file_name': 'train/c.jpg'}]}


def test_complete_partial_and_unselected_coverage_are_stat_only(tmp_path):
    (tmp_path / 'train').mkdir()
    (tmp_path / 'train/a.jpg').write_bytes(b'not even a real image')
    (tmp_path / 'train/b.jpg').write_bytes(b'also not an image')
    result, complete = inventory_paths(metadata(), {1, 2}, tmp_path)
    assert complete == {1}
    assert result['annotated_images'] == 3 and result['present_images'] == 2
    assert result['missing_images'] == 1 and result['partial_image_videos'] == 1
    assert result['present_bytes'] == 38
    assert result['pixel_bytes_hashed_or_decoded'] is result['dense_frame_coverage_certified'] is False
    assert result['missing_image_bytes_known'] is False
    selected, complete = inventory_paths(metadata(), {1}, tmp_path)
    assert selected['annotated_images'] == 1 and complete == {1}


@pytest.mark.parametrize('path', ('/absolute.jpg', '../val/x.jpg', 'val/x.jpg'))
def test_escaping_or_nontrain_metadata_path_refused(tmp_path, path):
    data = metadata()
    data['images'][0]['file_name'] = path
    with pytest.raises(ValueError):
        inventory_paths(data, {1}, tmp_path)


def test_symlink_escape_and_duplicate_paths_refused(tmp_path):
    (tmp_path / 'train').mkdir()
    (tmp_path / 'outside.jpg').write_bytes(b'x')
    (tmp_path / 'train/a.jpg').symlink_to(tmp_path / 'outside.jpg')
    with pytest.raises(ValueError, match='symlink escapes'):
        inventory_paths(metadata(), {1}, tmp_path)
    data = metadata()
    data['images'].append(data['images'][1].copy())
    with pytest.raises(ValueError, match='Duplicate'):
        inventory_paths(data, {2}, tmp_path)


def test_zero_size_nonregular_and_no_image_video_do_not_pass(tmp_path):
    (tmp_path / 'train').mkdir()
    (tmp_path / 'train/a.jpg').write_bytes(b'')
    result, complete = inventory_paths(metadata(), {1}, tmp_path)
    assert result['invalid_existing_files'] == 1 and not complete
    data = metadata()
    data['images'] = []
    with pytest.raises(ValueError):
        inventory_paths(data, {1}, tmp_path)
