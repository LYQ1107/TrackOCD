import json
import pytest
from src.trackocd_cvpr.metadata_only import image_video_metadata


def test_annotations_and_category_shuffle_are_not_decoded(tmp_path, monkeypatch):
    original = json.loads
    decoded = []
    def record(value, *args, **kwargs):
        decoded.append(value)
        return original(value, *args, **kwargs)
    # Synthetic fixtures contain no real dataset labels.
    first = tmp_path / 'a.json'
    second = tmp_path / 'b.json'
    first.write_text('{"annotations":[{"category":"DO_NOT_DECODE","x":"\\\"}[]"}],"images":[{"id":1}],"videos":[{"id":2}],"categories":[3]}')
    second.write_text('{"categories":[9],"videos":[{"id":2}],"images":[{"id":1}],"annotations":[{"category":"CHANGED_FUTURE_LABEL"}]}')
    monkeypatch.setattr(json, 'loads', record)
    assert image_video_metadata(first) == image_video_metadata(second)
    assert all(b'DO_NOT_DECODE' not in v and b'CHANGED_FUTURE_LABEL' not in v for v in decoded)


@pytest.mark.parametrize('text', ['{"images":[],"videos":[]', '{"images":[],"images":[],"videos":[]}', '[]'])
def test_malformed_metadata_rejected(tmp_path, text):
    path = tmp_path / 'bad.json'
    path.write_text(text)
    with pytest.raises((ValueError, IndexError)):
        image_video_metadata(path)
