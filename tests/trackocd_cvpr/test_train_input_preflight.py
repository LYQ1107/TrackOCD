import pytest
from scripts.trackocd_cvpr.prepare_train_input_counterfactual import train_image


def test_pixel_paths_are_Train_only(tmp_path):
    (tmp_path / 'train').mkdir()
    image = tmp_path / 'train' / 'fixture.jpg'
    image.write_bytes(b'synthetic path-only fixture')
    assert train_image(tmp_path, 'train/fixture.jpg') == image
    for relative in ('val/a.jpg', 'test/a.jpg', '../train/fixture.jpg', str(image)):
        with pytest.raises(ValueError):
            train_image(tmp_path, relative)


def test_links_cannot_escape_Train_tree(tmp_path):
    (tmp_path / 'train').mkdir()
    other = tmp_path / 'other.jpg'
    other.write_bytes(b'path fixture')
    (tmp_path / 'train' / 'link.jpg').symlink_to(other)
    with pytest.raises(ValueError):
        train_image(tmp_path, 'train/link.jpg')
