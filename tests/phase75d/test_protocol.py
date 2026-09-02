import csv
import numpy as np

from src.iclr27_phase75d.protocol import CSV_PATH, FEAT_PATH, PREFIXES


def test_frozen_protocol_shape_and_prefix_order():
    # Keep the contract test lightweight: the formal audit loads the aligned
    # arrays once; pytest should not mmap/decompress the 768-D cache repeatedly.
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        assert sum(1 for _ in csv.DictReader(f)) == 43423
    z = np.load(FEAT_PATH, allow_pickle=False)
    assert z["cls"].shape == (43423, 768)
    assert z["roi"].shape == (43423, 768)
    assert PREFIXES == (1, 2, 4, 8, 16)
    assert z["row_keys"].shape[0] == 43423
    assert np.isfinite(z["cls"][:8]).all() and np.isfinite(z["roi"][:8]).all()
