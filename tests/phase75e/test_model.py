import torch

from src.iclr27_phase75e.model import LowRankFeatureAdapter


def test_rank8_zero_up_projection_is_raw_preserving():
    torch.manual_seed(7501)
    x = torch.randn(5, 768)
    model = LowRankFeatureAdapter()
    y = model(x)
    raw = torch.nn.functional.normalize(x, dim=-1)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.allclose(y, raw, atol=1e-6, rtol=1e-6)
    assert model.A.weight.shape == (8, 768)
    assert model.B.weight.shape == (768, 8)


def test_adapter_has_gradient_through_selected_residual():
    torch.manual_seed(7502)
    model = LowRankFeatureAdapter()
    # Give B a small non-zero value only for this gradient test.
    with torch.no_grad():
        model.B.weight.normal_(std=1e-3)
    x = torch.randn(3, 768, requires_grad=False)
    out = model(x)
    loss = (out[:, :8] ** 2).mean()
    loss.backward()
    assert model.A.weight.grad is not None
    assert model.B.weight.grad is not None
    assert torch.isfinite(model.A.weight.grad).all()
