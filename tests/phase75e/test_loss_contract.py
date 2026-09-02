import torch

from src.iclr27_phase75e.losses import episode_loss
from src.iclr27_phase75e.model import LowRankFeatureAdapter


def test_fixed_loss_is_finite_and_raw_anchor_is_present():
    torch.manual_seed(7504)
    model = LowRankFeatureAdapter()
    q = {p: torch.randn(min(p, 4), 768) for p in (1, 2, 4, 8, 16)}
    pos = [{p: torch.randn(min(p + 1, 5), 768) for p in q}]
    neg = {p: torch.randn(min(p + 2, 6), 768) for p in q}
    loss, parts = episode_loss(model, q, pos, neg)
    assert torch.isfinite(loss)
    assert set(("rank", "raw_reconstruction", "safe")) <= set(parts)
    loss.backward()
    assert torch.isfinite(model.A.weight.grad).all()
