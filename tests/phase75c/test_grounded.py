import numpy as np

from scripts.iclr27_phase75c.run_r_retrieval import prefix_consistency
from src.iclr27_phase75c.grounded_correspondence import GroundedCorrespondence


def test_prefix_consistency_is_per_query_diagonal():
    identity = np.eye(3, dtype=np.float32)
    assert prefix_consistency({"1": identity, "2": identity, "4": identity, "8": identity, "16": identity}) == {
        "1": 1.0,
        "2": 1.0,
        "4": 1.0,
        "8": 1.0,
    }


def test_grounded_prefix_is_causal_and_normalized():
    rng = np.random.default_rng(75)
    x = rng.normal(size=(4, 768)).astype(np.float32)
    y = x.copy()
    y[1:] = rng.normal(size=(3, 768)).astype(np.float32)
    model = GroundedCorrespondence()
    assert np.array_equal(model.encode_prefix(x, 1), model.encode_prefix(y, 1))
    assert np.isclose(np.linalg.norm(model.encode_prefix(x, 4)), 1.0, atol=1e-5)
