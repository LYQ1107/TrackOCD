import numpy as np

from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks


def test_frozen_protocol_shape_and_prefix_order():
    table = load_frozen_tracks()
    assert len(table.rows) == 43423
    assert table.features.shape == (43423, 768)
    assert PREFIXES == (1, 2, 4, 8, 16)
    key = next(iter(table.sequences))
    assert table.get_frame_sequence(key, 1).shape[0] == 1
    assert table.get_frame_sequence(key, 16).shape[0] <= 16
    assert np.isfinite(table.features).all()
