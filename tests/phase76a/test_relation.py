import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.iclr27_phase76a.correspondence import hungarian_match, pair_relation_features, relation_summary
from src.iclr27_phase76a.raw_anchor import raw_mean_cosine, raw_mean_vector
from src.iclr27_phase76a.relation_model import AnchoredRelationReranker


def test_raw_anchor_matches_manual_normalize_mean():
    q = np.arange(1536, dtype=np.float32).reshape(2, 768) + 1.0
    c = np.flip(q, axis=1).copy()
    def norm(x): return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    expected = norm(norm(q).mean(0, keepdims=True))[0] @ norm(norm(c).mean(0, keepdims=True))[0]
    assert abs(raw_mean_cosine(q, c) - float(expected)) <= 1e-7
    assert np.isclose(np.linalg.norm(raw_mean_vector(q)), 1.0, atol=1e-6)


def test_hungarian_relation_shapes_and_summary():
    q = np.zeros((2, 768), dtype=np.float32); c = np.zeros((3, 768), dtype=np.float32)
    q[0, 0] = 1; q[1, 1] = 1; c[0, 0] = 1; c[1, 1] = 1; c[2, 2] = 1
    m = hungarian_match(q, c); f = pair_relation_features(q, c, m); s = relation_summary(q, c, m, 0.5)
    assert f.shape == (2, 1536)
    assert s.shape == (13,)
    assert np.isfinite(s).all()


def test_step_zero_is_raw_anchor():
    torch.manual_seed(7); model = AnchoredRelationReranker().eval()
    tokens = torch.randn(3, 1536); summary = torch.randn(13); raw = torch.tensor(0.37)
    with torch.no_grad(): y = model(tokens, summary, raw)
    assert abs(float(y['delta']) - 0.0) <= 1e-7
    assert abs(float(y['confidence']) - 0.5) <= 1e-7
    assert abs(float(y['final']) - float(raw)) <= 1e-7
