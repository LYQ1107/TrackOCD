import numpy as np

from src.iclr27_phase75d.pairwise_correspondence import hungarian_score


def test_query_and_support_suffixes_do_not_change_prefix_score():
    rng = np.random.default_rng(7505); q = rng.normal(size=(8, 768)).astype(np.float32); c = rng.normal(size=(16, 768)).astype(np.float32)
    q2 = q.copy(); q2[4:] = rng.normal(size=(4, 768)).astype(np.float32)
    c2 = c.copy(); c2[8:] = rng.normal(size=(8, 768)).astype(np.float32)
    assert np.isclose(hungarian_score(q[:4], c[:16]), hungarian_score(q2[:4], c2[:16]), atol=1e-7)
