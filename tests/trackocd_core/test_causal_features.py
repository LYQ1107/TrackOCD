from dataclasses import fields

import numpy as np

from src.trackocd_core.features import PREFIXES, PrefixView, prefix_view


def test_every_registered_prefix_is_independent_of_future_descriptors_and_geometry():
    rng = np.random.default_rng(1027)
    visual = rng.normal(size=(32, 768)).astype(np.float32)
    geometry = rng.random((32, 4), dtype=np.float32)
    quality = rng.random(32, dtype=np.float32)
    frames = np.arange(32) * 30
    for p in PREFIXES:
        before = prefix_view(visual, geometry, quality, frames, p)
        changed = [a.copy() for a in (visual, geometry, quality, frames)]
        for a in changed[:3]:
            a[p:] = np.nan  # A read of future values would fail validation.
        changed[3][p:] = -1
        after = prefix_view(*changed, p)
        np.testing.assert_array_equal(before.weighted_mean(), after.weighted_mean())
        np.testing.assert_array_equal(before.boxes_normalized_xyxy, after.boxes_normalized_xyxy)
        assert len(after.visual) == p


def test_prefix_view_does_not_expose_ids_labels_total_length_or_mutable_source():
    visual = np.ones((16, 768), dtype=np.float16)
    view = prefix_view(visual, np.ones((16, 4)), np.ones(16), np.arange(16), 2)
    assert {f.name for f in fields(PrefixView)} == {"visual", "boxes_normalized_xyxy", "quality", "elapsed_frames"}
    visual[:2] = 0
    assert view.visual.sum() == 2 * 768
    assert not view.visual.flags.writeable
    assert view.weighted_mean().shape == (768,)
    np.testing.assert_allclose(np.linalg.norm(view.weighted_mean()), 1, atol=1e-6)
