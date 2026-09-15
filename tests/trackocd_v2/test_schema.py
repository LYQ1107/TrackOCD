import numpy as np
import pytest

from src.trackocd_v2.schema import TrackSample


def test_prefix_mean_is_causal_and_excludes_labels():
    sample = TrackSample(
        sample_key="1_2",
        video_id=1,
        physical_track_id="2",
        frame_ids=[10, 20],
        boxes_xyxy=[[0, 0, 1, 1], [0, 0, 2, 2]],
        visual_features=np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        quality=[1, 1],
        gt_category_id=99,
        gt_split="new",
    )
    assert np.allclose(sample.prefix_feature(1), [1.0, 0.0])
    assert np.allclose(sample.full_feature(), [1.0, 1.0] / np.sqrt(2))
    assert "gt_category_id" not in sample.model_view()
    assert "gt_split" not in sample.model_view()


def test_prefix_cannot_read_future_rows():
    sample = TrackSample("1_2", 1, "2", [1], [[0, 0, 1, 1]], np.ones((1, 3), np.float32), [1])
    with pytest.raises(ValueError):
        sample.prefix_feature(2)
