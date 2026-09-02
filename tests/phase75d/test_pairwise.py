import numpy as np

from src.iclr27_phase75d.pairwise_correspondence import hungarian_score, pairwise_similarity


def test_hungarian_recovers_permuted_frames():
    a = np.zeros((2, 768), dtype=np.float32); b = np.zeros_like(a)
    a[0, 0] = 1.0; a[1, 1] = 1.0; b[:] = a[::-1]
    score = hungarian_score(a, b)
    direct = float(np.mean(np.sum(a * b, axis=1)))
    assert score > direct
    assert np.isclose(score, 1.0, atol=1e-6)


def test_candidate_permutation_invariance():
    rng = np.random.default_rng(7504); q = rng.normal(size=(4, 768)).astype(np.float32); c = rng.normal(size=(6, 768)).astype(np.float32)
    assert np.isclose(hungarian_score(q, c), hungarian_score(q, c[[3, 0, 5, 1, 4, 2]]), atol=1e-7)
    assert pairwise_similarity(q, c).shape == (4, 6)
